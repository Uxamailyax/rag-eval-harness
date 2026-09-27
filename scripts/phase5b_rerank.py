"""Phase 5b: does reranking help, and does it depend on chunk size?

5a established that `fixed` chunking wins at every size, and that 512 tokens is
the best of the three. This holds chunking fixed at that winning strategy and adds
one variable: the cross-encoder reranker.

**Why this is a separate stage rather than a wider 5a grid.** With the reranker
running during the sweep, a configuration could win because its chunking suits the
reranker rather than because the chunking is good. The conclusion would read
"fixed @ 512 is best" when the truth might be "the reranker likes long chunks."
One variable at a time keeps the attribution clean.

**Why three sizes and not just the winner.** A cross-encoder reads query and chunk
together under a 512-token limit. At 512-token chunks it sees roughly one whole
chunk per judgement; at 128 it sees the chunk with room to spare. More or less
context per judgement could genuinely change its accuracy, so the interaction is
worth measuring rather than assuming away.

**The hypothesis being tested.** The LegalBench-RAG authors reported that Cohere's
`rerank-english-v3.0` *degraded* retrieval on this benchmark — their RCTS recall
fell from 62.22% to 61.06% at K=64, and precision fell across the board. Their
explanation was that a general-purpose model does not align with legal text.

This is not a foregone conclusion here: different reranker (BGE vs Cohere),
different chunk sizes, different first stage. Reproducing their finding with a
different model would strengthen it considerably. Contradicting it would be
equally interesting.

**What the reranker cannot do.** It only reorders the 50 candidates the first stage
returned. A gold chunk ranked 60th is unreachable no matter how good the reranker
is. So the ceiling for this experiment is recall@50 of the first stage, and that is
reported alongside, because a reranker that "fails" against an already-exhausted
candidate pool has not really been tested.

Run with:  uv run python scripts/phase5b_rerank.py
           uv run python scripts/phase5b_rerank.py --sizes 512    (single size)
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
from rag_eval_harness.retrieval.rerank import CrossEncoderReranker, rank_movement

STRATEGY = "fixed"  # 5a's winner
SIZES = (128, 256, 512)
DEV_SUBSET_SIZE = 500
TOP_K = 10
CANDIDATE_K = 50
KS = (5, 10)
METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")
PREAMBLE_THRESHOLD = 200
N_EXAMPLES = 3


@dataclass
class Row:
    size: int
    reranked: bool
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    preamble_rank1: float
    rerank_ms_p50: float | None = None
    first_stage_ms_p50: float | None = None


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


def load_chunks(size: int) -> list[Chunk]:
    path = settings.cache_path.parent / f"chunks_{STRATEGY}_{size}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"chunk cache missing: {path.name}\nRun scripts/phase5_sweep.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def snippet(text: str, limit: int = 110) -> str:
    return " ".join(text[:limit].split()) + ("..." if len(text) > limit else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()

    source = LegalBenchRAG(settings.data_dir)
    dev = stratified_sample(
        source.usable_queries(), DEV_SUBSET_SIZE, settings.random_seed
    )

    print(f"strategy   : {STRATEGY} (5a winner)")
    print(f"sizes      : {args.sizes}")
    print(f"queries    : {len(dev)}")
    print(f"candidates : {CANDIDATE_K} -> top {TOP_K}\n")

    reranker = CrossEncoderReranker()
    print(f"reranker   : {reranker.model_name} on {reranker.device}\n")

    rows: list[Row] = []
    ceilings: dict[int, float] = {}
    examples: list[dict] = []

    for size in args.sizes:
        print(f"[{STRATEGY} @ {size}]")
        chunks = load_chunks(size)
        dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
        stats = dense.build(show_progress=False)
        if not stats.from_cache:
            print("  warning: embeddings rebuilt rather than loaded from cache")
        print(f"  {len(chunks)} chunks")

        base_scores: list[dict[str, float]] = []
        rr_scores: list[dict[str, float]] = []
        base_lat: list[float] = []
        rr_lat: list[float] = []
        rerank_only_lat: list[float] = []
        base_preamble = rr_preamble = 0
        ceiling_scores: list[dict[str, float]] = []
        movement_totals = {"promoted": 0, "demoted": 0, "unchanged": 0, "new_in_top_k": 0}

        for n, query in enumerate(dev, start=1):
            gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]

            # One first-stage call serves both arms: the top 10 of these candidates
            # is the no-rerank result, and all 50 are what the reranker sees. Running
            # retrieval twice would add noise to the latency comparison for nothing.
            t0 = time.perf_counter()
            candidates = dense.retrieve(query.text, top_k=CANDIDATE_K)
            first_ms = (time.perf_counter() - t0) * 1000
            base_lat.append(first_ms)

            baseline = candidates[:TOP_K]
            base_scores.append(
                evaluate_query(
                    [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in baseline],
                    gold,
                    ks=KS,
                )
            )
            if baseline and baseline[0].chunk.start <= PREAMBLE_THRESHOLD:
                base_preamble += 1

            # The ceiling: what recall would be if the reranker ordered all 50
            # perfectly. A reranker cannot exceed this, so a poor result against an
            # exhausted candidate pool is not the reranker's failure.
            ceiling_scores.append(
                evaluate_query(
                    [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in candidates],
                    gold,
                    ks=(CANDIDATE_K,),
                )
            )

            reranked, rr_ms = reranker.rerank(query.text, candidates, top_k=TOP_K)
            rerank_only_lat.append(rr_ms)
            rr_lat.append(first_ms + rr_ms)

            rr_scores.append(
                evaluate_query(
                    [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in reranked],
                    gold,
                    ks=KS,
                )
            )
            if reranked and reranked[0].chunk.start <= PREAMBLE_THRESHOLD:
                rr_preamble += 1

            for key, value in rank_movement(baseline, reranked).items():
                movement_totals[key] += value

            # Keep the clearest cases where reranking changed the outcome.
            delta = rr_scores[-1]["recall@10"] - base_scores[-1]["recall@10"]
            if abs(delta) > 0.5 and len(examples) < 40:
                examples.append(
                    {
                        "size": size,
                        "query_id": query.query_id,
                        "query": query.text,
                        "delta": round(delta, 4),
                        "base_recall": round(base_scores[-1]["recall@10"], 4),
                        "rr_recall": round(rr_scores[-1]["recall@10"], 4),
                        "base_top1_start": baseline[0].chunk.start if baseline else None,
                        "rr_top1_start": reranked[0].chunk.start if reranked else None,
                        "rr_top1_text": snippet(reranked[0].chunk.text) if reranked else "",
                    }
                )

            if n % 100 == 0:
                print(f"    {n}/{len(dev)}")

        ceilings[size] = aggregate(ceiling_scores)[f"recall@{CANDIDATE_K}"]

        rows.append(
            Row(
                size=size,
                reranked=False,
                metrics=aggregate(base_scores),
                p50_ms=percentile(base_lat, 0.5),
                p95_ms=percentile(base_lat, 0.95),
                preamble_rank1=base_preamble / len(dev),
            )
        )
        rows.append(
            Row(
                size=size,
                reranked=True,
                metrics=aggregate(rr_scores),
                p50_ms=percentile(rr_lat, 0.5),
                p95_ms=percentile(rr_lat, 0.95),
                preamble_rank1=rr_preamble / len(dev),
                rerank_ms_p50=percentile(rerank_only_lat, 0.5),
                first_stage_ms_p50=percentile(base_lat, 0.5),
            )
        )

        base_r = rows[-2].metrics["recall@10"]
        rr_r = rows[-1].metrics["recall@10"]
        arrow = "up" if rr_r > base_r else "DOWN"
        print(f"  off: recall@10 {base_r:.4f}   on: {rr_r:.4f}   ({arrow} "
              f"{abs(rr_r - base_r):.4f})")
        print(f"  ceiling recall@{CANDIDATE_K}: {ceilings[size]:.4f}")
        print(f"  movement: {movement_totals}\n")

    # --- results -------------------------------------------------------------
    print("=" * 88)
    print(f"PHASE 5b — RERANKER ON/OFF, {STRATEGY} chunks, {len(dev)} queries")
    print("=" * 88)
    print(f"\n  {'size':>6}{'rerank':>9}" + "".join(f"{m:>11}" for m in METRICS)
          + f"{'preamble':>10}{'p50 ms':>10}")
    print("  " + "-" * 86)
    for row in rows:
        cells = "".join(f"{row.metrics[f'{m}@10']:>11.4f}" for m in METRICS)
        print(f"  {row.size:>6}{'on' if row.reranked else 'off':>9}{cells}"
              f"{row.preamble_rank1:>9.1%}{row.p50_ms:>10.1f}")

    print(f"\n{'=' * 88}")
    print("DID RERANKING HELP?")
    print("=" * 88)
    print(f"\n  {'size':>6}{'off':>10}{'on':>10}{'delta':>10}{'headroom':>11}"
          f"{'used':>9}{'+latency':>11}")
    print("  " + "-" * 67)

    deltas: dict[int, float] = {}
    for size in args.sizes:
        off = next(r for r in rows if r.size == size and not r.reranked)
        on = next(r for r in rows if r.size == size and r.reranked)
        delta = on.metrics["recall@10"] - off.metrics["recall@10"]
        deltas[size] = delta

        # Headroom is what reranking could gain if it ordered all 50 perfectly.
        # "used" is the fraction of that headroom actually captured — the honest
        # way to judge a reranker, because a small delta against small headroom is
        # a different result from a small delta against large headroom.
        headroom = ceilings[size] - off.metrics["recall@10"]
        used = delta / headroom if headroom > 0 else 0.0
        added = (on.rerank_ms_p50 or 0.0)

        print(f"  {size:>6}{off.metrics['recall@10']:>10.4f}"
              f"{on.metrics['recall@10']:>10.4f}{delta:>+10.4f}"
              f"{headroom:>11.4f}{used:>8.1%}{added:>10.1f}ms")

    print("\n  headroom = recall@50 of the first stage minus recall@10 without")
    print("  reranking. This is the most a perfect reranker could recover by")
    print("  reordering the same 50 candidates.")

    helped = sum(1 for d in deltas.values() if d > 0.005)
    hurt = sum(1 for d in deltas.values() if d < -0.005)

    print("\n  verdict:")
    if hurt == len(args.sizes):
        print("  -> Reranking hurt at every chunk size. This reproduces the")
        print("     LegalBench-RAG authors' finding with a different reranker")
        print("     (BGE rather than Cohere), which strengthens it.")
    elif helped == len(args.sizes):
        print("  -> Reranking helped at every chunk size, contradicting the")
        print("     benchmark authors' Cohere result. The difference is the model.")
    else:
        print("  -> Mixed: reranking helps at some chunk sizes and not others.")
        print("     Chunk size and reranking interact, which is why this was")
        print("     tested across sizes rather than only on the 5a winner.")

    if examples:
        print(f"\n{'=' * 88}")
        print("QUERIES WHERE RERANKING CHANGED THE OUTCOME")
        print("=" * 88)
        gained = sorted([e for e in examples if e["delta"] > 0],
                        key=lambda e: -e["delta"])[:N_EXAMPLES]
        lost = sorted([e for e in examples if e["delta"] < 0],
                      key=lambda e: e["delta"])[:N_EXAMPLES]

        for title, group in (("RERANKING RECOVERED", gained), ("RERANKING LOST", lost)):
            if not group:
                continue
            print(f"\n  {title}")
            for e in group:
                print(f"\n    [{e['query_id']}] size {e['size']}  "
                      f"{e['base_recall']:.2f} -> {e['rr_recall']:.2f}")
                print(f"      Q: {snippet(e['query'], 130)}")
                print(f"      top1 moved: char {e['base_top1_start']} -> "
                      f"{e['rr_top1_start']}")
                print(f"      new top1: {e['rr_top1_text']}")

    # --- gate ----------------------------------------------------------------
    checks = {
        "every size scored both arms": len(rows) == len(args.sizes) * 2,
        "reranking changed the ranking": any(
            abs(d) > 1e-6 for d in deltas.values()
        ),
        "reranked recall never exceeds the candidate ceiling": all(
            next(r for r in rows if r.size == s and r.reranked).metrics["recall@10"]
            <= ceilings[s] + 1e-9
            for s in args.sizes
        ),
    }

    print(f"\n{'=' * 88}")
    print("--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You can say whether you would ship the reranker, and why")
    print("  [ ] You can explain the latency cost in proportion to the gain")

    manifest_path = write_manifest(
        settings,
        {
            "phase": "5b",
            "experiment": "reranker_on_off",
            "strategy": STRATEGY,
            "sizes": args.sizes,
            "reranker": reranker.model_name,
            "device": reranker.device,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "dev_subset_size": len(dev),
            "ceilings": {str(k): round(v, 6) for k, v in ceilings.items()},
            "deltas": {str(k): round(v, 6) for k, v in deltas.items()},
            "automated_checks": checks,
            "rows": [asdict(r) for r in rows],
        },
    )

    (manifest_path.parent / "rerank_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
