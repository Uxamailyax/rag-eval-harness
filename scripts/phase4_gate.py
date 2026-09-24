"""Phase 4 gate: BM25 vs dense vs hybrid.

Same 500 queries, same chunks, same metrics as Phase 3. Only the search method
changes, so any difference in the numbers is attributable to the method.

Beyond the comparison table this answers two specific questions:

  1. Does dense retrieval also return the document preamble? Phase 3 found BM25
     ranking the character-0 chunk first in 4 of 5 inspected queries, because the
     party names in the query outweigh the actual question. The embedding of that
     same query is also dominated by the party names, so dense may land in the same
     place for a different reason. If it does, the problem is the query format
     rather than lexical matching, and the Phase 5 prefix-stripping experiment
     becomes central rather than a footnote.

  2. Where do BM25 and dense disagree, and why? A per-query comparison surfaces the
     queries each method wins, which is what makes the aggregate numbers
     explainable rather than merely reportable.

Run with:  uv run python scripts/phase4_gate.py
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
from rag_eval_harness.retrieval.dense import DenseRetriever
from rag_eval_harness.retrieval.hybrid import HybridRetriever

CHUNK_SIZE_TOKENS = 256
DEV_SUBSET_SIZE = 500
TOP_K = 10
CANDIDATE_K = 50
KS = (5, 10)
METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")

# A chunk starting within this many characters of the document start counts as the
# preamble: title, parties, dates, recitals.
PREAMBLE_THRESHOLD = 200


def stratified_sample(queries: list[Query], size: int, seed: int) -> list[Query]:
    """Sample proportionally across subsets. Identical to Phase 3, same seed, so
    the two phases score exactly the same queries."""
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


def build_chunks(source: LegalBenchRAG, chunker: FixedSizeChunker) -> list[Chunk]:
    """Reuse the Phase 3 chunk cache. Same chunks, or the comparison is invalid."""
    tag = f"{chunker.tokenizer_name.replace('/', '_')}_{chunker.chunk_size_tokens}"
    cache_path = settings.cache_path.parent / f"chunks_{tag}.pkl"

    if cache_path.exists():
        print(f"  loading chunks from {cache_path.name}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print("  no chunk cache found, tokenising (several minutes)...")
    chunks: list[Chunk] = []
    for n, doc_id in enumerate(source.document_ids(), start=1):
        try:
            text = source.document_text(doc_id)
        except FileNotFoundError:
            continue
        chunks.extend(chunker.chunk(doc_id, text))
        if n % 100 == 0:
            print(f"    {n} documents, {len(chunks)} chunks")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(chunks, fh)
    return chunks


def show(label: str, text: str, limit: int = 150) -> None:
    body = " ".join(text[:limit].split())
    print(f"        {label}: {body}{'...' if len(text) > limit else ''}")


def main() -> int:
    source = LegalBenchRAG(settings.data_dir)
    usable = source.usable_queries()
    dev = stratified_sample(usable, DEV_SUBSET_SIZE, settings.random_seed)

    print("--- setup ---")
    print(f"  dev subset : {len(dev)} queries (seed={settings.random_seed})")
    print(f"  per subset : {source.subset_counts(dev)}")

    chunker = FixedSizeChunker(chunk_size_tokens=CHUNK_SIZE_TOKENS)
    chunks = build_chunks(source, chunker)
    print(f"  chunks     : {len(chunks)}")

    print("\n--- building indexes ---")
    t0 = time.perf_counter()
    bm25 = BM25Retriever(chunks)
    print(f"  bm25  : {time.perf_counter() - t0:.1f}s")

    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
    print(f"  dense : embedding on {dense.device}")
    stats = dense.build()
    source_label = "cache" if stats.from_cache else f"{stats.seconds:.0f}s"
    print(f"  dense : {stats.n_chunks} x {stats.dim} vectors ({source_label})")

    hybrid = HybridRetriever(
        {"bm25": bm25, "dense": dense}, candidate_k=CANDIDATE_K
    )

    retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    # --- run every retriever over every query ------------------------------
    print(f"\n--- retrieving top-{TOP_K} for {len(dev)} queries x 3 retrievers ---")
    per_query: dict[str, list[dict[str, float]]] = {n: [] for n in retrievers}
    latencies: dict[str, list[float]] = {n: [] for n in retrievers}
    preamble_first: dict[str, int] = {n: 0 for n in retrievers}
    preamble_any: dict[str, int] = {n: 0 for n in retrievers}
    rows: list[dict] = []

    for n, query in enumerate(dev, start=1):
        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]
        row: dict = {
            "query_id": query.query_id,
            "subset": query.subset,
            "query_text": query.text,
        }

        for name, retriever in retrievers.items():
            t0 = time.perf_counter()
            hits = retriever.retrieve(query.text, top_k=TOP_K)
            latencies[name].append((time.perf_counter() - t0) * 1000)

            retrieved = [
                CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits
            ]
            scores = evaluate_query(retrieved, gold, ks=KS)
            per_query[name].append(scores)

            if hits and hits[0].chunk.start <= PREAMBLE_THRESHOLD:
                preamble_first[name] += 1
            if any(h.chunk.start <= PREAMBLE_THRESHOLD for h in hits):
                preamble_any[name] += 1

            row[name] = {
                "recall@10": scores["recall@10"],
                "mrr@10": scores["mrr@10"],
                "top1_start": hits[0].chunk.start if hits else None,
                "top1_doc": hits[0].chunk.doc_id if hits else None,
            }

        rows.append(row)
        if n % 50 == 0:
            print(f"  {n}/{len(dev)}")

    macro = {name: aggregate(scores) for name, scores in per_query.items()}

    # --- comparison table ---------------------------------------------------
    print(f"\n{'=' * 74}")
    print(f"PHASE 4 — BM25 vs dense vs hybrid, {len(dev)} queries, fixed 256-tok chunks")
    print("=" * 74)

    for k in KS:
        print(f"\n  @{k}")
        print(f"  {'retriever':<12}" + "".join(f"{m:>12}" for m in METRICS))
        print(f"  {'-' * 72}")
        for name in ("bm25", "dense", "hybrid"):
            cells = "".join(f"{macro[name][f'{m}@{k}']:>12.4f}" for m in METRICS)
            print(f"  {name:<12}{cells}")

    print(f"\n  {'retriever':<12}{'p50 ms':>10}{'p95 ms':>10}")
    print(f"  {'-' * 32}")
    lat_summary = {}
    for name in ("bm25", "dense", "hybrid"):
        vals = sorted(latencies[name])
        p50 = vals[len(vals) // 2]
        p95 = vals[int(len(vals) * 0.95)]
        lat_summary[name] = {"p50": round(p50, 2), "p95": round(p95, 2)}
        print(f"  {name:<12}{p50:>10.1f}{p95:>10.1f}")

    # --- the preamble question ----------------------------------------------
    print(f"\n{'=' * 74}")
    print("PREAMBLE CHECK — does dense fall for the title page too?")
    print("=" * 74)
    print(f"\n  A chunk starting within {PREAMBLE_THRESHOLD} chars of the document start")
    print("  is counted as preamble (title, parties, dates, recitals).\n")
    print(f"  {'retriever':<12}{'rank 1 is preamble':>22}{'preamble in top 10':>22}")
    print(f"  {'-' * 56}")
    preamble_summary = {}
    for name in ("bm25", "dense", "hybrid"):
        pf = preamble_first[name] / len(dev)
        pa = preamble_any[name] / len(dev)
        preamble_summary[name] = {"rank1": round(pf, 4), "any": round(pa, 4)}
        print(f"  {name:<12}{pf:>21.1%}{pa:>22.1%}")

    if preamble_summary["dense"]["rank1"] > 0.3 and preamble_summary["bm25"]["rank1"] > 0.3:
        print("\n  -> Both fall for it. The problem is the query format, not lexical")
        print("     matching. Prefix stripping moves to the centre of Phase 5.")
    elif preamble_summary["dense"]["rank1"] < preamble_summary["bm25"]["rank1"] / 2:
        print("\n  -> Dense largely avoids it. Semantic matching is doing real work")
        print("     that keyword matching could not.")

    # --- where they disagree -------------------------------------------------
    print(f"\n{'=' * 74}")
    print("DISAGREEMENT — queries where one method clearly beats the other")
    print("=" * 74)

    diffs = [
        (r["bm25"]["recall@10"] - r["dense"]["recall@10"], r) for r in rows
    ]
    diffs.sort(key=lambda t: t[0])

    def report(title: str, items: list[tuple[float, dict]], winner: str) -> None:
        print(f"\n  {title}")
        for delta, r in items:
            other = "dense" if winner == "bm25" else "bm25"
            print(f"\n    [{r['query_id']}] {winner} {r[winner]['recall@10']:.2f} "
                  f"vs {other} {r[other]['recall@10']:.2f}")
            print(f"      Q: {r['query_text'][:140]}")
            print(f"      {winner:<7} top1 @char {r[winner]['top1_start']}")
            print(f"      {other:<7} top1 @char {r[other]['top1_start']}")

    report("DENSE WINS BIGGEST:", diffs[:3], "dense")
    report("BM25 WINS BIGGEST:", diffs[-3:][::-1], "bm25")

    # --- gate ---------------------------------------------------------------
    best = max(("bm25", "dense", "hybrid"), key=lambda n: macro[n]["recall@10"])

    checks = {
        "all three retrievers scored every query": all(
            len(per_query[n]) == len(dev) for n in retrievers
        ),
        "dense recall@10 is non-zero": macro["dense"]["recall@10"] > 0,
        "hybrid found candidates from both components": any(
            d.found_by_both for d in hybrid.last_details
        ),
        "recall@10 >= recall@5 for all three": all(
            macro[n]["recall@10"] >= macro[n]["recall@5"] - 1e-9 for n in retrievers
        ),
    }

    print(f"\n{'=' * 74}")
    print("--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\n  best recall@10: {best} ({macro[best]['recall@10']:.4f})")
    print(f"  vs Phase 3 BM25 baseline: {macro['bm25']['recall@10']:.4f}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You can explain each gap between the three rows out loud")
    print("  [ ] You can explain where hybrid loses, if it does")
    print("  [ ] The disagreement examples above make sense when you read them")

    manifest_path = write_manifest(
        settings,
        {
            "phase": 4,
            "chunker": "fixed",
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "embedding_model": dense.model_name,
            "device": dense.device,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": hybrid.k,
            "dev_subset_size": len(dev),
            "total_chunks": len(chunks),
            "macro": {
                n: {k: round(v, 6) for k, v in m.items()} for n, m in macro.items()
            },
            "latency_ms": lat_summary,
            "preamble": preamble_summary,
            "best_recall_at_10": best,
            "automated_checks": checks,
        },
    )

    with open(manifest_path.parent / "per_query.jsonl", "w", encoding="utf-8") as fh:
        for i, query in enumerate(dev):
            fh.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "subset": query.subset,
                        **{
                            f"{name}_{k}": per_query[name][i][k]
                            for name in retrievers
                            for k in per_query[name][i]
                        },
                    }
                )
                + "\n"
            )

    (manifest_path.parent / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
