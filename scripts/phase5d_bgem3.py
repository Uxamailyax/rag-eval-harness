"""Phase 5d: does the conclusion survive a bigger embedder, and does the trend continue?

Two questions, one run.

**1. Is `fixed @ 512` a property of the corpus, or of bge-small?**

Every retrieval number in this project came from `BAAI/bge-small-en-v1.5` — 33M
parameters, 384 dimensions. If the winning configuration only wins on that model,
the finding is about the model rather than about chunking. Re-running the winner on
`BAAI/bge-m3` (568M parameters, 1024 dimensions) tests whether it holds.

**2. Did the recall trend stop at 512, or did the model stop?**

5a measured recall climbing steadily with chunk size: 0.18 → 0.29 → 0.42 at
128 / 256 / 512. The sweep stopped at 512 because that is bge-small's maximum
sequence length, not because the trend flattened. Anything longer is silently
truncated — the back half of a 1024-token chunk is invisible to the embedder while
remaining fully visible to BM25, which would produce a confident and wrong
conclusion that large chunks hurt dense retrieval.

BGE-M3 accepts 8,192 tokens, so it can actually read a 1024-token chunk. Running
both models at both sizes separates the two explanations:

    bge-small @ 512   the 5a winner, baseline
    bge-small @ 1024  truncated at 512 — included deliberately to measure the damage
    bge-m3    @ 512   same chunks, bigger model: does the winner hold?
    bge-m3    @ 1024  the honest test of whether the trend continues

The bge-small @ 1024 row is not a mistake. It quantifies what silent truncation
costs, which is the kind of thing that is usually assumed rather than measured.

**Cost.** BGE-M3 is 2.2 GB to download and produces 1024-dimensional vectors, so
embedding is slower and the cache files are ~2.7x larger. Expect roughly 30 minutes
per configuration on a GTX 1650.

Run with:  uv run python scripts/phase5d_bgem3.py
           uv run python scripts/phase5d_bgem3.py --sizes 512   (skip the 1024 runs)
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
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.metrics.retrieval import CharRange, aggregate, evaluate_query
from rag_eval_harness.retrieval.dense import DenseRetriever

SMALL = "BAAI/bge-small-en-v1.5"
LARGE = "BAAI/bge-m3"

MODEL_LIMITS = {SMALL: 512, LARGE: 8192}

STRATEGY = "fixed"
SIZES = (512, 1024)
DEV_SUBSET_SIZE = 500
TOP_K = 10
KS = (5, 10)
METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")
PREAMBLE_THRESHOLD = 200

# BGE-M3 vectors are 1024-dim rather than 384, so a batch holds ~2.7x more memory.
# 4 GB of VRAM does not tolerate the same batch size.
BATCH = {SMALL: 64, LARGE: 16}


@dataclass
class Row:
    model: str
    size: int
    truncated: bool
    n_chunks: int
    dim: int
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    preamble_rank1: float
    embed_seconds: float


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


def build_chunks(source: LegalBenchRAG, size: int) -> list[Chunk]:
    """Chunk at `size` tokens, reusing the 5a cache where it exists.

    Chunking always uses bge-small's tokenizer regardless of which model will embed
    the result. That is deliberate: the two models must see identical chunk
    boundaries or the comparison measures tokenization differences as well as model
    differences.
    """
    cache_path = settings.cache_path.parent / f"chunks_{STRATEGY}_{size}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  chunking at {size} tokens (not cached)...")
    chunker = FixedSizeChunker(chunk_size_tokens=size, tokenizer_name=SMALL)
    chunks: list[Chunk] = []
    doc_ids = source.document_ids()
    for n, doc_id in enumerate(doc_ids, start=1):
        try:
            text = source.document_text(doc_id)
        except FileNotFoundError:
            continue
        chunks.extend(chunker.chunk(doc_id, text))
        if n % 200 == 0:
            print(f"    {n}/{len(doc_ids)} docs, {len(chunks)} chunks")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(chunks, fh)
    return chunks


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def score(retriever: DenseRetriever, dev: list[Query]) -> tuple[dict, float, float, float]:
    per_query: list[dict[str, float]] = []
    latencies: list[float] = []
    preamble = 0

    for query in dev:
        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]
        t0 = time.perf_counter()
        hits = retriever.retrieve(query.text, top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]
        per_query.append(evaluate_query(retrieved, gold, ks=KS))
        if hits and hits[0].chunk.start <= PREAMBLE_THRESHOLD:
            preamble += 1

    return (
        aggregate(per_query),
        percentile(latencies, 0.5),
        percentile(latencies, 0.95),
        preamble / len(dev),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--models", nargs="+", default=[SMALL, LARGE])
    args = parser.parse_args()

    source = LegalBenchRAG(settings.data_dir)
    dev = stratified_sample(
        source.usable_queries(), DEV_SUBSET_SIZE, settings.random_seed
    )

    print(f"strategy : {STRATEGY}")
    print(f"sizes    : {args.sizes}")
    print(f"models   : {[m.split('/')[-1] for m in args.models]}")
    print(f"queries  : {len(dev)}\n")
    print("  model limits: bge-small 512 tokens, bge-m3 8192 tokens")
    print("  chunks above a model's limit are silently truncated by that model\n")

    rows: list[Row] = []

    for size in args.sizes:
        print(f"[chunks @ {size} tokens]")
        chunks = build_chunks(source, size)
        mean_len = sum(c.length for c in chunks) / len(chunks)
        print(f"  {len(chunks)} chunks, mean {mean_len:.0f} chars\n")

        for model in args.models:
            truncated = size > MODEL_LIMITS[model]
            flag = "  [TRUNCATED at "f"{MODEL_LIMITS[model]}]" if truncated else ""
            print(f"  {model.split('/')[-1]} @ {size}{flag}")

            dense = DenseRetriever(
                chunks,
                model_name=model,
                cache_dir=settings.cache_path.parent,
                batch_size=BATCH.get(model, 32),
            )
            stats = dense.build(show_progress=True)
            src = "cache" if stats.from_cache else f"{stats.seconds:.0f}s"
            print(f"    {stats.n_chunks} x {stats.dim}-dim vectors ({src})")

            metrics, p50, p95, preamble = score(dense, dev)
            rows.append(
                Row(
                    model=model,
                    size=size,
                    truncated=truncated,
                    n_chunks=len(chunks),
                    dim=stats.dim,
                    metrics={k: round(v, 6) for k, v in metrics.items()},
                    p50_ms=round(p50, 2),
                    p95_ms=round(p95, 2),
                    preamble_rank1=round(preamble, 4),
                    embed_seconds=round(stats.seconds, 1),
                )
            )
            print(f"    recall@10 {metrics['recall@10']:.4f}   "
                  f"ndcg@10 {metrics['ndcg@10']:.4f}   "
                  f"prec@10 {metrics['precision@10']:.4f}   "
                  f"preamble {preamble:.1%}   p50 {p50:.1f}ms\n")

    # --- results -------------------------------------------------------------
    print("=" * 98)
    print(f"PHASE 5d — EMBEDDER COMPARISON, {STRATEGY} chunks, {len(dev)} queries")
    print("=" * 98)
    print(f"\n  {'model':<16}{'size':>6}{'dim':>6}{'chunks':>9}"
          + "".join(f"{m:>11}" for m in METRICS) + f"{'preamble':>10}{'p50 ms':>9}")
    print("  " + "-" * 96)
    for r in rows:
        name = r.model.split("/")[-1]
        mark = " T" if r.truncated else "  "
        cells = "".join(f"{r.metrics[f'{m}@10']:>11.4f}" for m in METRICS)
        print(f"  {name:<16}{r.size:>6}{r.dim:>6}{r.n_chunks:>9}{cells}"
              f"{r.preamble_rank1:>9.1%}{r.p50_ms:>9.1f}{mark}")
    print("\n  T = the chunk exceeds this model's maximum sequence length and is")
    print("      truncated: the model never sees the tail of the chunk.")

    def find(model: str, size: int) -> Row | None:
        return next((r for r in rows if r.model == model and r.size == size), None)

    # --- question 1: does the winner hold on a bigger model? ------------------
    print(f"\n{'=' * 98}")
    print("QUESTION 1 — is `fixed @ 512` a property of the corpus or of bge-small?")
    print("=" * 98)

    s512, l512 = find(SMALL, 512), find(LARGE, 512)
    if s512 and l512:
        delta = l512.metrics["recall@10"] - s512.metrics["recall@10"]
        rel = delta / s512.metrics["recall@10"]
        print(f"\n  identical chunks, different embedder:\n")
        print(f"    bge-small (33M,  384-dim): recall@10 {s512.metrics['recall@10']:.4f}"
              f"   ndcg {s512.metrics['ndcg@10']:.4f}   p50 {s512.p50_ms:.1f}ms")
        print(f"    bge-m3    (568M, 1024-dim): recall@10 {l512.metrics['recall@10']:.4f}"
              f"   ndcg {l512.metrics['ndcg@10']:.4f}   p50 {l512.p50_ms:.1f}ms")
        print(f"\n    delta: {delta:+.4f} ({rel:+.1%} relative) for 17x the parameters")

        if abs(rel) < 0.10:
            print("\n  -> The 17x larger model moves recall by less than 10%. The")
            print("     configuration finding is about the corpus and the chunking,")
            print("     not about the embedder. bge-small was not the bottleneck.")
        elif rel > 0:
            print("\n  -> The larger model is materially better. Model size was a real")
            print("     constraint, and the earlier numbers understate what this corpus")
            print("     allows.")
        else:
            print("\n  -> The larger model is worse, which is worth investigating before")
            print("     reporting: check the query prefix and normalisation settings.")

    # --- question 2: did the trend stop, or did the model? -------------------
    print(f"\n{'=' * 98}")
    print("QUESTION 2 — did the recall trend stop at 512, or did bge-small stop?")
    print("=" * 98)
    print("\n  5a measured recall rising with chunk size on bge-small:")
    print("    128 -> 0.1767    256 -> 0.2931    512 -> 0.4226")
    print("  The sweep stopped at 512 because that is the model's limit, not")
    print("  because the trend flattened.\n")

    s1024, l1024 = find(SMALL, 1024), find(LARGE, 1024)
    if l512 and l1024:
        d = l1024.metrics["recall@10"] - l512.metrics["recall@10"]
        print(f"    bge-m3 @  512: recall@10 {l512.metrics['recall@10']:.4f}   "
              f"precision {l512.metrics['precision@10']:.4f}")
        print(f"    bge-m3 @ 1024: recall@10 {l1024.metrics['recall@10']:.4f}   "
              f"precision {l1024.metrics['precision@10']:.4f}")
        print(f"\n    delta: {d:+.4f}")

        prec_d = l1024.metrics["precision@10"] - l512.metrics["precision@10"]
        if d > 0.01 and prec_d < 0:
            print("\n  -> Recall keeps rising past 512, but precision falls again. The")
            print("     trend continues, and continues to be partly the size artifact")
            print("     from 5a: bigger chunks catch gold by covering more ground.")
        elif d > 0.01:
            print("\n  -> Recall keeps rising past 512 and precision holds. The 512")
            print("     ceiling was the model's limit, not the corpus's.")
        else:
            print("\n  -> Recall flattens past 512. The trend genuinely stops there,")
            print("     independently of the model limit.")

    # --- the cost of silent truncation ---------------------------------------
    if s1024 and l1024:
        print(f"\n{'=' * 98}")
        print("THE COST OF SILENT TRUNCATION")
        print("=" * 98)
        gap = l1024.metrics["recall@10"] - s1024.metrics["recall@10"]
        print(f"\n  Identical 1024-token chunks, both models:\n")
        print(f"    bge-small (reads first 512 tokens only): "
              f"recall@10 {s1024.metrics['recall@10']:.4f}")
        print(f"    bge-m3    (reads all 1024):              "
              f"recall@10 {l1024.metrics['recall@10']:.4f}")
        print(f"\n    difference: {gap:+.4f}")
        print("\n  This is what silent truncation costs, measured rather than assumed.")
        print("  No error is raised and no warning appears — the model simply never")
        print("  sees the tail of the chunk, and the resulting number looks entirely")
        print("  plausible. This is why 5a stopped at 512.")

    # --- gate ----------------------------------------------------------------
    checks = {
        "every configuration scored non-zero recall": all(
            r.metrics["recall@10"] > 0 for r in rows
        ),
        "bge-m3 produced 1024-dim vectors": all(
            r.dim == 1024 for r in rows if r.model == LARGE
        ),
        "both models scored the identical chunk sets": len(
            {(r.size, r.n_chunks) for r in rows}
        ) == len(args.sizes),
    }

    print(f"\n{'=' * 98}")
    print("--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You can say whether the 5a conclusion depends on the embedder")
    print("  [ ] You can say whether a bigger embedder is worth its cost here")

    manifest_path = write_manifest(
        settings,
        {
            "phase": "5d",
            "experiment": "embedder_comparison",
            "strategy": STRATEGY,
            "sizes": args.sizes,
            "models": args.models,
            "model_limits": MODEL_LIMITS,
            "dev_subset_size": len(dev),
            "automated_checks": checks,
            "rows": [asdict(r) for r in rows],
        },
    )

    with open(manifest_path.parent / "embedder_comparison.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
