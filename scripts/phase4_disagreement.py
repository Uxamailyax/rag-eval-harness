"""Phase 4 disagreement analysis, done properly.

The gate script reported only each retriever's rank-1 chunk, which is usually not
the chunk that earned the recall. That made the disagreement examples unreadable:
two retrievers could both show "top1 @char 0" while one scored 1.00 and the other
0.00, because the scoring chunk was at rank 4 and never printed.

This reports, for each retriever, the **highest-ranked chunk that actually overlaps
gold** — the chunk responsible for the score — alongside where gold sits and what
the rank-1 chunk was. That makes a disagreement explainable rather than merely
visible.

Reuses the cached chunks and embeddings, so it runs in seconds rather than the hour
the first embedding pass took.

Run with:  uv run python scripts/phase4_disagreement.py
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict

from rag_eval_harness.chunking.fixed import Chunk, FixedSizeChunker
from rag_eval_harness.config import settings
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.metrics.retrieval import CharRange, evaluate_query
from rag_eval_harness.retrieval.bm25 import BM25Retriever, RetrievedChunk
from rag_eval_harness.retrieval.dense import DenseRetriever

CHUNK_SIZE_TOKENS = 256
DEV_SUBSET_SIZE = 500
TOP_K = 10
N_EXAMPLES = 4
PREAMBLE_THRESHOLD = 200


def stratified_sample(queries: list[Query], size: int, seed: int) -> list[Query]:
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


def load_chunks(chunker: FixedSizeChunker) -> list[Chunk]:
    tag = f"{chunker.tokenizer_name.replace('/', '_')}_{chunker.chunk_size_tokens}"
    path = settings.cache_path.parent / f"chunks_{tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"chunk cache missing: {path}\nRun scripts/phase3_gate.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def first_scoring_hit(
    hits: list[RetrievedChunk], gold: list[CharRange]
) -> tuple[RetrievedChunk | None, int]:
    """Highest-ranked chunk that overlaps gold, and how many characters it covers.

    This is the chunk that produced the score. Reporting rank 1 instead is what
    made the original output unreadable.
    """
    for hit in hits:
        chunk_range = CharRange(hit.chunk.doc_id, hit.chunk.start, hit.chunk.end)
        covered = sum(chunk_range.overlap(g) for g in gold)
        if covered > 0:
            return hit, covered
    return None, 0


def snippet(text: str, limit: int = 130) -> str:
    return " ".join(text[:limit].split()) + ("..." if len(text) > limit else "")


def describe(
    name: str,
    hits: list[RetrievedChunk],
    gold: list[CharRange],
    recall: float,
    source: LegalBenchRAG,
) -> None:
    print(f"\n    {name.upper():<6} recall@10 = {recall:.2f}")

    if hits:
        top = hits[0]
        tag = " (preamble)" if top.chunk.start <= PREAMBLE_THRESHOLD else ""
        print(f"      rank 1     : [{top.chunk.start}:{top.chunk.end}]{tag}"
              f"  {top.chunk.doc_id.split('/')[-1][:50]}")

    hit, covered = first_scoring_hit(hits, gold)
    if hit is None:
        in_gold_doc = sum(1 for h in hits if any(h.chunk.doc_id == g.doc_id for g in gold))
        print(f"      scoring    : none of the 10 chunks touch gold")
        print(f"      note       : {in_gold_doc}/10 chunks were from the right document")
    else:
        gold_total = sum(g.length for g in gold)
        print(f"      scoring    : rank {hit.rank}  [{hit.chunk.start}:{hit.chunk.end}]"
              f"  covers {covered}/{gold_total} gold chars")
        print(f"      text       : {snippet(hit.chunk.text)}")


def main() -> int:
    source = LegalBenchRAG(settings.data_dir)
    dev = stratified_sample(
        source.usable_queries(), DEV_SUBSET_SIZE, settings.random_seed
    )

    chunker = FixedSizeChunker(chunk_size_tokens=CHUNK_SIZE_TOKENS)
    chunks = load_chunks(chunker)
    print(f"chunks: {len(chunks)}   queries: {len(dev)}\n")

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
    stats = dense.build(show_progress=False)
    if not stats.from_cache:
        print("warning: embeddings were rebuilt, not loaded from cache")

    print("scoring...")
    records = []
    for query in dev:
        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]
        bm25_hits = bm25.retrieve(query.text, top_k=TOP_K)
        dense_hits = dense.retrieve(query.text, top_k=TOP_K)

        bm25_recall = evaluate_query(
            [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in bm25_hits],
            gold,
            ks=(TOP_K,),
        )[f"recall@{TOP_K}"]
        dense_recall = evaluate_query(
            [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in dense_hits],
            gold,
            ks=(TOP_K,),
        )[f"recall@{TOP_K}"]

        records.append(
            {
                "query": query,
                "gold": gold,
                "bm25_hits": bm25_hits,
                "dense_hits": dense_hits,
                "bm25_recall": bm25_recall,
                "dense_recall": dense_recall,
                "delta": dense_recall - bm25_recall,
            }
        )

    records.sort(key=lambda r: r["delta"], reverse=True)

    for title, subset in (
        ("DENSE WINS — semantic matching found what keywords missed", records[:N_EXAMPLES]),
        ("BM25 WINS — exact terms beat semantic similarity", records[-N_EXAMPLES:][::-1]),
    ):
        print(f"\n{'=' * 78}")
        print(title)
        print("=" * 78)

        for rec in subset:
            query = rec["query"]
            gold = rec["gold"]
            print(f"\n  [{query.query_id}]  delta = {rec['delta']:+.2f}")
            print(f"  Q: {snippet(query.text, 160)}")

            gold_total = sum(g.length for g in gold)
            print(f"\n    GOLD ({len(gold)} span(s), {gold_total} chars)")
            for g in gold[:2]:
                text = source.document_text(g.doc_id)[g.start : g.end]
                print(f"      [{g.start}:{g.end}]  {snippet(text, 110)}")

            describe("bm25", rec["bm25_hits"], gold, rec["bm25_recall"], source)
            describe("dense", rec["dense_hits"], gold, rec["dense_recall"], source)
            print("  " + "-" * 74)

    # --- aggregate picture of where the two differ -------------------------
    both_hit = sum(1 for r in records if r["bm25_recall"] > 0 and r["dense_recall"] > 0)
    only_bm25 = sum(1 for r in records if r["bm25_recall"] > 0 and r["dense_recall"] == 0)
    only_dense = sum(1 for r in records if r["bm25_recall"] == 0 and r["dense_recall"] > 0)
    neither = sum(1 for r in records if r["bm25_recall"] == 0 and r["dense_recall"] == 0)

    print(f"\n{'=' * 78}")
    print("OVERLAP — which queries each method solves")
    print("=" * 78)
    n = len(records)
    print(f"\n  both found gold      : {both_hit:>4}  ({both_hit / n:.1%})")
    print(f"  only bm25 found gold : {only_bm25:>4}  ({only_bm25 / n:.1%})")
    print(f"  only dense found gold: {only_dense:>4}  ({only_dense / n:.1%})")
    print(f"  neither              : {neither:>4}  ({neither / n:.1%})")
    print(f"\n  union (either found something): {(n - neither) / n:.1%}")
    print("  This is the ceiling hybrid fusion can reach at top-10: fusion can only")
    print("  reorder what the components already returned, never find something new.")

    out = settings.results_dir / "phase4_disagreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "both_hit": both_hit,
                "only_bm25": only_bm25,
                "only_dense": only_dense,
                "neither": neither,
                "union_rate": round((n - neither) / n, 4),
                "examples": [
                    {
                        "query_id": r["query"].query_id,
                        "query": r["query"].text,
                        "delta": round(r["delta"], 4),
                        "bm25_recall": round(r["bm25_recall"], 4),
                        "dense_recall": round(r["dense_recall"], 4),
                    }
                    for r in records[:N_EXAMPLES] + records[-N_EXAMPLES:]
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
