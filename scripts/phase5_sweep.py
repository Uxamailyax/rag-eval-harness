"""Phase 5 sweep: which settings actually matter.

Everything so far ran on one fixed configuration — 256-token chunks, cut at
arbitrary positions, no overlap. Nothing has established whether those were good
choices. This changes one variable at a time and measures.

    4 chunking strategies x 3 sizes x 2 retrievers = 24 rows

Both retrievers are included because chunk size may affect them differently. BM25
scores on term density, so a larger chunk holds more terms and may match more
readily; dense retrieval compresses a whole chunk into one vector, so a larger
chunk dilutes the signal. If those two pull in opposite directions, "the best chunk
size" is not one number, and that is worth knowing before Phase 6 fixes a
configuration.

**Why every configuration is re-chunked and re-embedded.** Chunk boundaries change
with the strategy and the size, so the embeddings built for one configuration
describe text that does not exist in another. Reusing them would score one setup
using another's vectors. Each configuration is cached separately and the cache key
includes the strategy, the size, and a fingerprint of the chunk set.

**A note on what this can and cannot show.** Each row varies one thing against a
fixed baseline, so interactions are not measured — whether paragraph chunking
specifically prefers 512 tokens, for instance. That is an honest limitation of a
grid this size, and it belongs in the write-up rather than being papered over.

Run with:  uv run python scripts/phase5_sweep.py
           uv run python scripts/phase5_sweep.py --sizes 256    (quick check)
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

from rag_eval_harness.chunking.fixed import Chunk, FixedSizeChunker
from rag_eval_harness.chunking.semantic import SemanticChunker
from rag_eval_harness.chunking.structural import (
    ParagraphChunker,
    RecursiveChunker,
    _TokenCounter,
)
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.metrics.retrieval import CharRange, aggregate, evaluate_query
from rag_eval_harness.retrieval.bm25 import BM25Retriever
from rag_eval_harness.retrieval.dense import DenseRetriever

SIZES = (128, 256, 512)
STRATEGIES = ("fixed", "recursive", "paragraph", "semantic")
DEV_SUBSET_SIZE = 500
TOP_K = 10
KS = (5, 10)
METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")
PREAMBLE_THRESHOLD = 200


@dataclass
class ConfigResult:
    strategy: str
    size: int
    retriever: str
    n_chunks: int
    mean_chunk_chars: float
    median_chunk_chars: float
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    preamble_rank1: float
    chunk_seconds: float
    embed_seconds: float


def stratified_sample(queries: list[Query], size: int, seed: int) -> list[Query]:
    """Same sample as Phases 3 and 4. Identical seed, so every phase scores the
    same queries and the numbers are comparable across phases."""
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


def make_chunker(strategy: str, size: int, counter: _TokenCounter):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size_tokens=size)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size_tokens=size, counter=counter)
    if strategy == "paragraph":
        return ParagraphChunker(chunk_size_tokens=size, counter=counter)
    if strategy == "semantic":
        return SemanticChunker(chunk_size_tokens=size, counter=counter)
    raise ValueError(f"unknown strategy: {strategy}")


def build_chunks(
    source: LegalBenchRAG, strategy: str, size: int, counter: _TokenCounter
) -> tuple[list[Chunk], float]:
    """Chunk the corpus for one configuration, caching to disk."""
    cache_path = settings.cache_path.parent / f"chunks_{strategy}_{size}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh), 0.0

    chunker = make_chunker(strategy, size, counter)
    doc_ids = source.document_ids()
    chunks: list[Chunk] = []

    started = time.perf_counter()
    for n, doc_id in enumerate(doc_ids, start=1):
        try:
            text = source.document_text(doc_id)
        except FileNotFoundError:
            continue
        chunks.extend(chunker.chunk(doc_id, text))
        if n % 200 == 0:
            print(f"      {n}/{len(doc_ids)} docs, {len(chunks)} chunks", flush=True)
    elapsed = time.perf_counter() - started

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(chunks, fh)
    return chunks, elapsed


def score_retriever(
    retriever: object, dev: list[Query], label: str
) -> tuple[dict[str, float], float, float, float]:
    """Run one retriever over the dev set. Returns metrics, p50, p95, preamble rate."""
    per_query: list[dict[str, float]] = []
    latencies: list[float] = []
    preamble_first = 0

    for query in dev:
        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]

        t0 = time.perf_counter()
        hits = retriever.retrieve(query.text, top_k=TOP_K)  # type: ignore[attr-defined]
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]
        per_query.append(evaluate_query(retrieved, gold, ks=KS))

        if hits and hits[0].chunk.start <= PREAMBLE_THRESHOLD:
            preamble_first += 1

    latencies.sort()
    return (
        aggregate(per_query),
        latencies[len(latencies) // 2],
        latencies[int(len(latencies) * 0.95)],
        preamble_first / len(dev),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    args = parser.parse_args()

    source = LegalBenchRAG(settings.data_dir)
    dev = stratified_sample(
        source.usable_queries(), DEV_SUBSET_SIZE, settings.random_seed
    )

    total_configs = len(args.strategies) * len(args.sizes)
    print(f"dev subset  : {len(dev)} queries")
    print(f"strategies  : {args.strategies}")
    print(f"sizes       : {args.sizes}")
    print(f"retrievers  : {args.retrievers}")
    print(f"configs     : {total_configs} chunkings x {len(args.retrievers)} retrievers "
          f"= {total_configs * len(args.retrievers)} rows\n")

    # One shared token counter across all chunkers: its cache is the difference
    # between tokenising a 300 KB merger agreement once and doing it dozens of
    # times over.
    counter = _TokenCounter()
    results: list[ConfigResult] = []
    config_n = 0

    for strategy in args.strategies:
        for size in args.sizes:
            config_n += 1
            print(f"[{config_n}/{total_configs}] {strategy} @ {size} tokens")

            chunks, chunk_secs = build_chunks(source, strategy, size, counter)
            lengths = sorted(c.length for c in chunks)
            mean_len = sum(lengths) / len(lengths)
            median_len = lengths[len(lengths) // 2]
            tag = "cached" if chunk_secs == 0 else f"{chunk_secs:.0f}s"
            print(f"  chunks: {len(chunks)}  mean {mean_len:.0f} chars  "
                  f"median {median_len} chars  ({tag})")

            embed_secs = 0.0
            for name in args.retrievers:
                if name == "bm25":
                    t0 = time.perf_counter()
                    retriever: object = BM25Retriever(chunks)
                    build_secs = time.perf_counter() - t0
                else:
                    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
                    stats = dense.build(show_progress=True)
                    embed_secs = stats.seconds
                    build_secs = stats.seconds
                    retriever = dense

                metrics, p50, p95, preamble = score_retriever(retriever, dev, name)
                results.append(
                    ConfigResult(
                        strategy=strategy,
                        size=size,
                        retriever=name,
                        n_chunks=len(chunks),
                        mean_chunk_chars=round(mean_len, 1),
                        median_chunk_chars=float(median_len),
                        metrics={k: round(v, 6) for k, v in metrics.items()},
                        p50_ms=round(p50, 2),
                        p95_ms=round(p95, 2),
                        preamble_rank1=round(preamble, 4),
                        chunk_seconds=round(chunk_secs, 1),
                        embed_seconds=round(embed_secs, 1),
                    )
                )
                print(f"    {name:<6} recall@10 {metrics['recall@10']:.4f}  "
                      f"ndcg@10 {metrics['ndcg@10']:.4f}  "
                      f"prec@10 {metrics['precision@10']:.4f}  "
                      f"preamble {preamble:.1%}  ({build_secs:.0f}s index)")
            print()

    # --- results table ------------------------------------------------------
    print("=" * 96)
    print("PHASE 5 — CHUNKING ABLATION")
    print("=" * 96)

    for name in args.retrievers:
        rows = [r for r in results if r.retriever == name]
        if not rows:
            continue
        print(f"\n  {name.upper()}\n")
        print(f"  {'strategy':<12}{'size':>6}{'chunks':>9}{'mean ch':>9}"
              f"{'recall@10':>11}{'prec@10':>10}{'mrr@10':>9}{'ndcg@10':>10}"
              f"{'preamble':>10}{'p50 ms':>9}")
        print("  " + "-" * 94)
        best = max(rows, key=lambda r: r.metrics["recall@10"])
        for r in sorted(rows, key=lambda r: (r.strategy, r.size)):
            mark = " *" if r is best else "  "
            print(f"  {r.strategy:<12}{r.size:>6}{r.n_chunks:>9}"
                  f"{r.mean_chunk_chars:>9.0f}{r.metrics['recall@10']:>11.4f}"
                  f"{r.metrics['precision@10']:>10.4f}{r.metrics['mrr@10']:>9.4f}"
                  f"{r.metrics['ndcg@10']:>10.4f}{r.preamble_rank1:>9.1%}"
                  f"{r.p50_ms:>9.1f}{mark}")

    # --- what varying each thing does ---------------------------------------
    print(f"\n{'=' * 96}")
    print("EFFECT OF EACH VARIABLE")
    print("=" * 96)

    for name in args.retrievers:
        rows = [r for r in results if r.retriever == name]
        if not rows:
            continue
        print(f"\n  {name.upper()} — mean recall@10 by size (across strategies)")
        for size in sorted(set(r.size for r in rows)):
            subset = [r for r in rows if r.size == size]
            avg = sum(r.metrics["recall@10"] for r in subset) / len(subset)
            avg_p = sum(r.metrics["precision@10"] for r in subset) / len(subset)
            print(f"    {size:>4} tokens   recall {avg:.4f}   precision {avg_p:.4f}")

        print(f"\n  {name.upper()} — mean recall@10 by strategy (across sizes)")
        for strategy in sorted(set(r.strategy for r in rows)):
            subset = [r for r in rows if r.strategy == strategy]
            avg = sum(r.metrics["recall@10"] for r in subset) / len(subset)
            avg_p = sum(r.metrics["precision@10"] for r in subset) / len(subset)
            print(f"    {strategy:<12} recall {avg:.4f}   precision {avg_p:.4f}")

    # --- do the two retrievers want the same thing? -------------------------
    if len(args.retrievers) > 1:
        print(f"\n{'=' * 96}")
        print("DO SPARSE AND DENSE WANT THE SAME CHUNKING?")
        print("=" * 96)
        bm25_rows = {(r.strategy, r.size): r for r in results if r.retriever == "bm25"}
        dense_rows = {(r.strategy, r.size): r for r in results if r.retriever == "dense"}
        if bm25_rows and dense_rows:
            best_bm25 = max(bm25_rows.values(), key=lambda r: r.metrics["recall@10"])
            best_dense = max(dense_rows.values(), key=lambda r: r.metrics["recall@10"])
            print(f"\n  best for bm25 : {best_bm25.strategy} @ {best_bm25.size}"
                  f"  (recall {best_bm25.metrics['recall@10']:.4f})")
            print(f"  best for dense: {best_dense.strategy} @ {best_dense.size}"
                  f"  (recall {best_dense.metrics['recall@10']:.4f})")
            if (best_bm25.strategy, best_bm25.size) == (best_dense.strategy, best_dense.size):
                print("\n  -> Same configuration wins for both. One chunking serves both.")
            else:
                print("\n  -> Different configurations win. 'The best chunk size' is not")
                print("     one number; it depends on the retriever.")

    # --- gate ---------------------------------------------------------------
    overall_best = max(results, key=lambda r: r.metrics["recall@10"])
    phase4_dense_recall = 0.2931

    checks = {
        "every configuration produced chunks": all(r.n_chunks > 0 for r in results),
        "every configuration scored non-zero recall": all(
            r.metrics["recall@10"] > 0 for r in results
        ),
        "chunk counts vary with size": len(set(r.n_chunks for r in results)) > 1,
        "at least one config beats the Phase 4 dense baseline": (
            overall_best.metrics["recall@10"] > phase4_dense_recall
        ),
    }

    print(f"\n{'=' * 96}")
    print("--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\n  best overall: {overall_best.strategy} @ {overall_best.size} "
          f"({overall_best.retriever})  recall@10 = "
          f"{overall_best.metrics['recall@10']:.4f}")
    print(f"  Phase 4 baseline (fixed @ 256, dense): {phase4_dense_recall:.4f}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You can name the configuration you would ship")
    print("  [ ] You can state the condition that would change your mind")
    print("  [ ] The trivial paragraph control did not quietly win unnoticed")

    manifest_path = write_manifest(
        settings,
        {
            "phase": 5,
            "sweep": "chunking_ablation",
            "strategies": args.strategies,
            "sizes": args.sizes,
            "retrievers": args.retrievers,
            "dev_subset_size": len(dev),
            "top_k": TOP_K,
            "best": {
                "strategy": overall_best.strategy,
                "size": overall_best.size,
                "retriever": overall_best.retriever,
                "recall@10": overall_best.metrics["recall@10"],
            },
            "automated_checks": checks,
            "results": [asdict(r) for r in results],
        },
    )

    with open(manifest_path.parent / "sweep.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
