"""Phase 3 gate: the first number that comes from retrieval rather than arithmetic.

One chunker (fixed, 256 tokens), one retriever (BM25), scored against the verified
ground truth from Phase 1 using the metrics verified in Phase 2.

Deliberately one of each. If this number looks wrong, there is exactly one place it
can be coming from — Phase 1 proved the ground truth reads correctly and Phase 2
proved the metrics compute correctly, so anything left is retrieval.

The corpus is indexed **whole**, not just the documents the dev queries touch.
Indexing only the relevant documents would mean every chunk in the index came from a
document containing an answer, which makes retrieval far easier than it is and
produces a number that cannot be compared to anything real.

Run with:  uv run python scripts/phase3_gate.py
"""

from __future__ import annotations

import json
import pickle
import random
import sys
import time
from collections import defaultdict

from rag_eval_harness.chunking.fixed import Chunk, FixedSizeChunker
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.metrics.retrieval import CharRange, aggregate, evaluate_query
from rag_eval_harness.retrieval.bm25 import BM25Retriever

CHUNK_SIZE_TOKENS = 256
DEV_SUBSET_SIZE = 500
TOP_K = 10
KS = (5, 10)
INSPECT_N = 5


# ---------------------------------------------------------------------------
# Dev subset
# ---------------------------------------------------------------------------


def stratified_sample(queries: list[Query], size: int, seed: int) -> list[Query]:
    """Sample `size` queries, proportional to each subset's share of the whole.

    Proportional rather than flat random because CUAD holds ~60% of the queries;
    a flat sample would let one subset's characteristics dominate the headline
    number while appearing to describe the whole benchmark.
    """
    by_subset: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by_subset[q.subset].append(q)

    rng = random.Random(seed)
    total = len(queries)
    sampled: list[Query] = []

    for subset in sorted(by_subset):
        pool = sorted(by_subset[subset], key=lambda q: q.query_id)
        take = min(len(pool), max(1, round(size * len(pool) / total)))
        sampled.extend(rng.sample(pool, take))

    rng.shuffle(sampled)
    return sampled[:size]


# ---------------------------------------------------------------------------
# Chunking, with a disk cache
# ---------------------------------------------------------------------------


def build_chunks(source: LegalBenchRAG, chunker: FixedSizeChunker) -> list[Chunk]:
    """Chunk every document in the corpus, caching the result to disk.

    Tokenising ~90 MB takes minutes. The cache key includes the chunk size and
    tokenizer name, so changing either correctly produces a new cache entry
    rather than silently reusing chunks built under different settings.
    """
    tag = f"{chunker.tokenizer_name.replace('/', '_')}_{chunker.chunk_size_tokens}"
    cache_path = settings.cache_path.parent / f"chunks_{tag}.pkl"

    if cache_path.exists():
        print(f"  loading cached chunks from {cache_path.name}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    doc_ids = source.document_ids()
    print(f"  tokenising {len(doc_ids)} documents (this takes a few minutes)...")

    chunks: list[Chunk] = []
    started = time.perf_counter()
    for n, doc_id in enumerate(doc_ids, start=1):
        try:
            text = source.document_text(doc_id)
        except FileNotFoundError:
            continue
        chunks.extend(chunker.chunk(doc_id, text))
        if n % 100 == 0:
            print(f"    {n}/{len(doc_ids)} documents, {len(chunks)} chunks")

    elapsed = time.perf_counter() - started
    print(f"  chunked in {elapsed:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(chunks, fh)
    print(f"  cached to {cache_path.name}")

    return chunks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def show_text(label: str, text: str, limit: int = 220) -> None:
    body = " ".join(text[:limit].split())
    suffix = "..." if len(text) > limit else ""
    print(f"      {label}: {body}{suffix}")


def print_inspection(rows: list[dict], source: LegalBenchRAG, n: int) -> None:
    """Print real text for a few queries. Reading these is the gate.

    A metrics table can be wrong in ways that look entirely reasonable. Reading
    the query, the gold answer and what was actually retrieved is the only check
    that catches a number that is arithmetically correct and substantively wrong.
    """
    print(f"\n{'=' * 70}")
    print(f"MANUAL INSPECTION — {n} queries. Read these before trusting the table.")
    print("=" * 70)

    for row in rows[:n]:
        print(f"\n[{row['query_id']}]  recall@10={row['recall@10']:.3f}  "
              f"precision@10={row['precision@10']:.3f}  mrr@10={row['mrr@10']:.3f}")
        print(f"  Q: {row['query_text'][:200]}")

        print(f"\n  GOLD ({row['n_gold_spans']} span(s), {row['gold_chars']} chars):")
        for span in row["gold_preview"]:
            print(f"    - {span['doc_id']}  [{span['start']}:{span['end']}]")
            show_text("text", span["text"])

        print(f"\n  RETRIEVED (top {min(3, len(row['retrieved_preview']))} of {TOP_K}):")
        for hit in row["retrieved_preview"][:3]:
            mark = "HIT " if hit["overlaps_gold"] else "miss"
            print(f"    {hit['rank']}. [{mark}] score={hit['score']:.2f}  "
                  f"{hit['doc_id']}  [{hit['start']}:{hit['end']}]")
            show_text("text", hit["text"], limit=160)
        print("-" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    source = LegalBenchRAG(settings.data_dir)

    print("--- ground truth ---")
    usable = source.usable_queries()
    print(f"  usable queries : {len(usable)}")
    print(f"  per subset     : {source.subset_counts(usable)}")

    dev = stratified_sample(usable, DEV_SUBSET_SIZE, settings.random_seed)
    print(f"\n  dev subset     : {len(dev)} queries (seed={settings.random_seed})")
    print(f"  per subset     : {source.subset_counts(dev)}")

    print(f"\n--- chunking (fixed, {CHUNK_SIZE_TOKENS} tokens, no overlap) ---")
    chunker = FixedSizeChunker(chunk_size_tokens=CHUNK_SIZE_TOKENS)
    chunks = build_chunks(source, chunker)
    total_chars = sum(c.length for c in chunks)
    print(f"  chunks     : {len(chunks)}")
    print(f"  mean length: {total_chars / len(chunks):.0f} chars")

    print("\n--- BM25 index ---")
    started = time.perf_counter()
    retriever = BM25Retriever(chunks)
    print(f"  built over {len(retriever)} chunks in {time.perf_counter() - started:.1f}s")

    print(f"\n--- retrieving top-{TOP_K} for {len(dev)} queries ---")
    per_query: list[dict[str, float]] = []
    rows: list[dict] = []
    latencies: list[float] = []

    for n, query in enumerate(dev, start=1):
        t0 = time.perf_counter()
        hits = retriever.retrieve(query.text, top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]
        # Every retrieved chunk is scored, including ones from documents with no
        # gold span. Filtering those out first would make irrelevant retrieval
        # free and inflate precision.
        retrieved = [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]

        scores = evaluate_query(retrieved, gold, ks=KS)
        per_query.append(scores)

        if len(rows) < INSPECT_N:
            gold_ids = {s.doc_id for s in query.spans}
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_text": query.text,
                    "n_gold_spans": len(query.spans),
                    "gold_chars": query.total_gold_chars,
                    **{k: scores[k] for k in ("recall@10", "precision@10", "mrr@10")},
                    "gold_preview": [
                        {
                            "doc_id": s.doc_id,
                            "start": s.start,
                            "end": s.end,
                            "text": source.document_text(s.doc_id)[s.start : s.end],
                        }
                        for s in query.spans[:2]
                    ],
                    "retrieved_preview": [
                        {
                            "rank": h.rank,
                            "score": h.score,
                            "doc_id": h.chunk.doc_id,
                            "start": h.chunk.start,
                            "end": h.chunk.end,
                            "text": h.chunk.text,
                            "overlaps_gold": any(
                                CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end).overlap(
                                    CharRange(s.doc_id, s.start, s.end)
                                )
                                > 0
                                for s in query.spans
                            ),
                            "in_gold_doc": h.chunk.doc_id in gold_ids,
                        }
                        for h in hits
                    ],
                }
            )

        if n % 100 == 0:
            print(f"  {n}/{len(dev)}")

    macro = aggregate(per_query)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\n{'=' * 70}")
    print(f"PHASE 3 — BM25, fixed {CHUNK_SIZE_TOKENS}-token chunks, {len(dev)} queries")
    print("=" * 70)
    print(f"\n  {'metric':<16}{'@5':>10}{'@10':>10}")
    print(f"  {'-' * 36}")
    for name in ("hit_rate", "recall", "precision", "mrr", "ndcg"):
        print(f"  {name:<16}{macro[f'{name}@5']:>10.4f}{macro[f'{name}@10']:>10.4f}")
    print(f"\n  retrieval latency: p50 {p50:.1f}ms   p95 {p95:.1f}ms")

    print_inspection(rows, source, INSPECT_N)

    checks = {
        "every query returned results": len(per_query) == len(dev),
        "recall@10 is non-zero": macro["recall@10"] > 0,
        "recall@10 is not suspiciously perfect": macro["recall@10"] < 0.999,
        "recall@10 >= recall@5 (more chunks cannot find less)": (
            macro["recall@10"] >= macro["recall@5"] - 1e-9
        ),
        "precision@5 >= precision@10 (later chunks are worse)": (
            macro["precision@5"] >= macro["precision@10"] - 1e-9
        ),
    }

    print("\n--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] The 5 inspected queries make sense: retrieved chunks that are")
    print("      marked HIT really do contain the gold text")
    print("  [ ] You can say in one sentence why the number is what it is")

    manifest_path = write_manifest(
        settings,
        {
            "phase": 3,
            "retriever": "bm25",
            "chunker": "fixed",
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "tokenizer": chunker.tokenizer_name,
            "top_k": TOP_K,
            "dev_subset_size": len(dev),
            "total_chunks": len(chunks),
            "subset_counts": source.subset_counts(dev),
            "macro": {k: round(v, 6) for k, v in macro.items()},
            "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2)},
            "automated_checks": checks,
        },
    )

    with open(manifest_path.parent / "per_query.jsonl", "w", encoding="utf-8") as fh:
        for query, scores in zip(dev, per_query):
            fh.write(
                json.dumps(
                    {"query_id": query.query_id, "subset": query.subset, **scores}
                )
                + "\n"
            )

    (manifest_path.parent / "inspection.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
