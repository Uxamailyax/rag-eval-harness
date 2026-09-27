"""Phase 5c: is the preamble problem caused by the query format?

Every query in this benchmark follows one template:

    Consider [document description]; [actual question]

The description is roughly 80% of the query text and consists almost entirely of
party names and the agreement type. Phase 4 measured both BM25 and dense placing a
document preamble at rank 1 about 54% of the time, and Phase 5a showed that rising
to 64% at 512-token chunks. The party names are densest in the preamble, so both
retrievers land there — by completely different mechanisms, with the same result.

This tests three arms on the shipping configuration:

    full      the query as written                          (5a baseline)
    stripped  everything after the semicolon only           (the experiment)
    oracle    full query, but search restricted to the
              document the gold span lives in               (the upper bound)

**The stripped arm has a real trade.** Remove the party names and "Is there a
non-compete clause?" matches non-compete clauses in all 714 contracts — nothing
identifies which one is meant. So the prefix is doing document selection badly,
through the same scoring channel as passage matching. Three outcomes are possible
and all are informative: recall up means the prefix hurt more than it helped, recall
down means it was doing necessary work, flat means the effects cancel and the real
fix is elsewhere.

**The oracle arm is deliberate cheating.** It uses the gold span's own document id to
filter, which no real system could do. That is the point: it measures how much
performance is lost to conflating document selection with passage retrieval, without
requiring a document classifier to be built first. It is reported as a diagnostic
upper bound, never as a result.

Run with:  uv run python scripts/phase5c_prefix.py
"""

from __future__ import annotations

import json
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

from rag_eval_harness.chunking.fixed import Chunk, FixedSizeChunker
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.metrics.retrieval import CharRange, aggregate, evaluate_query
from rag_eval_harness.retrieval.bm25 import RetrievedChunk
from rag_eval_harness.retrieval.dense import DenseRetriever

STRATEGY = "fixed"
SIZE = 512  # 5a winner
DEV_SUBSET_SIZE = 500
TOP_K = 10
KS = (5, 10)
METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")
PREAMBLE_THRESHOLD = 200
N_EXAMPLES = 3


@dataclass
class Row:
    arm: str
    metrics: dict[str, float]
    p50_ms: float
    p95_ms: float
    preamble_rank1: float
    right_doc_rank1: float
    mean_docs_in_top10: float


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


def strip_prefix(text: str) -> str:
    """Drop everything up to and including the first semicolon.

    No entity extraction or parsing: the template puts a literal semicolon between
    the document description and the question, so one split does it.
    """
    if ";" not in text:
        return text
    return text.split(";", 1)[1].strip()


def load_chunks() -> list[Chunk]:
    path = settings.cache_path.parent / f"chunks_{STRATEGY}_{SIZE}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"chunk cache missing: {path.name}\nRun scripts/phase5_sweep.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def snippet(text: str, limit: int = 120) -> str:
    return " ".join(text[:limit].split()) + ("..." if len(text) > limit else "")


class DocumentFilteredDense:
    """Dense retrieval restricted to one document's chunks.

    Implemented by masking the score vector rather than rebuilding an index, so the
    oracle arm costs one extra dot product rather than 500 index builds.
    """

    def __init__(self, dense: DenseRetriever, chunks: list[Chunk]):
        self.dense = dense
        self.chunks = chunks
        self._doc_index: dict[str, np.ndarray] = {}
        for i, chunk in enumerate(chunks):
            self._doc_index.setdefault(chunk.doc_id, []).append(i)  # type: ignore[arg-type]
        self._doc_index = {
            doc: np.asarray(idx, dtype=np.int64) for doc, idx in self._doc_index.items()
        }

    def retrieve(
        self, query: str, doc_ids: set[str], top_k: int = TOP_K
    ) -> list[RetrievedChunk]:
        parts = [self._doc_index[d] for d in doc_ids if d in self._doc_index]
        if not parts:
            return []
        rows = np.concatenate(parts)
        if rows.size == 0:
            return []

        q = self.dense.encode_query(query)
        scores = self.dense._embeddings[rows] @ q  # noqa: SLF001

        k = min(top_k, rows.size)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [
            RetrievedChunk(
                chunk=self.chunks[int(rows[i])], score=float(scores[i]), rank=rank
            )
            for rank, i in enumerate(top, start=1)
        ]


def main() -> int:
    source = LegalBenchRAG(settings.data_dir)
    dev = stratified_sample(
        source.usable_queries(), DEV_SUBSET_SIZE, settings.random_seed
    )

    print(f"config  : {STRATEGY} @ {SIZE}, dense (5a winner)")
    print(f"queries : {len(dev)}\n")

    # --- does the template hold? -------------------------------------------
    with_semicolon = [q for q in dev if ";" in q.text]
    multi = [q for q in dev if q.text.count(";") > 1]
    print("--- query template check ---")
    print(f"  contain a semicolon : {len(with_semicolon)}/{len(dev)} "
          f"({len(with_semicolon) / len(dev):.1%})")
    print(f"  more than one       : {len(multi)}")

    lengths_full = [len(q.text) for q in dev]
    lengths_stripped = [len(strip_prefix(q.text)) for q in dev]
    kept = sum(lengths_stripped) / sum(lengths_full)
    print(f"  mean query chars    : {np.mean(lengths_full):.0f} full -> "
          f"{np.mean(lengths_stripped):.0f} stripped")
    print(f"  text retained       : {kept:.1%} (so the prefix is "
          f"{1 - kept:.1%} of the query)\n")

    if len(with_semicolon) < len(dev):
        print(f"  note: {len(dev) - len(with_semicolon)} queries have no semicolon and")
        print("  are passed through unchanged in the stripped arm.\n")

    # --- setup --------------------------------------------------------------
    chunks = load_chunks()
    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
    stats = dense.build(show_progress=False)
    if not stats.from_cache:
        print("warning: embeddings rebuilt rather than loaded from cache")
    print(f"chunks  : {len(chunks)}\n")

    filtered = DocumentFilteredDense(dense, chunks)

    arms = ("full", "stripped", "oracle")
    per_query: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    latencies: dict[str, list[float]] = {a: [] for a in arms}
    preamble: dict[str, int] = {a: 0 for a in arms}
    right_doc: dict[str, int] = {a: 0 for a in arms}
    doc_spread: dict[str, list[int]] = {a: [] for a in arms}
    examples: list[dict] = []
    oracle_empty = 0

    print("--- running three arms ---")
    for n, query in enumerate(dev, start=1):
        gold = [CharRange(s.doc_id, s.start, s.end) for s in query.spans]
        gold_docs = {s.doc_id for s in query.spans}
        stripped_text = strip_prefix(query.text)

        results: dict[str, list[RetrievedChunk]] = {}

        t0 = time.perf_counter()
        results["full"] = dense.retrieve(query.text, top_k=TOP_K)
        latencies["full"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        results["stripped"] = dense.retrieve(stripped_text, top_k=TOP_K)
        latencies["stripped"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        results["oracle"] = filtered.retrieve(query.text, gold_docs, top_k=TOP_K)
        latencies["oracle"].append((time.perf_counter() - t0) * 1000)
        if not results["oracle"]:
            oracle_empty += 1

        for arm, hits in results.items():
            retrieved = [
                CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits
            ]
            per_query[arm].append(evaluate_query(retrieved, gold, ks=KS))

            if hits and hits[0].chunk.start <= PREAMBLE_THRESHOLD:
                preamble[arm] += 1
            if hits and hits[0].chunk.doc_id in gold_docs:
                right_doc[arm] += 1
            doc_spread[arm].append(len({h.chunk.doc_id for h in hits}))

        full_r = per_query["full"][-1]["recall@10"]
        strip_r = per_query["stripped"][-1]["recall@10"]
        if abs(strip_r - full_r) > 0.5 and len(examples) < 40:
            examples.append(
                {
                    "query_id": query.query_id,
                    "subset": query.subset,
                    "full_query": query.text,
                    "stripped_query": stripped_text,
                    "full_recall": round(full_r, 4),
                    "stripped_recall": round(strip_r, 4),
                    "delta": round(strip_r - full_r, 4),
                    "full_top1_doc": results["full"][0].chunk.doc_id if results["full"] else None,
                    "stripped_top1_doc": (
                        results["stripped"][0].chunk.doc_id if results["stripped"] else None
                    ),
                    "gold_doc": sorted(gold_docs)[0],
                    "full_docs_in_top10": len({h.chunk.doc_id for h in results["full"]}),
                    "stripped_docs_in_top10": len(
                        {h.chunk.doc_id for h in results["stripped"]}
                    ),
                }
            )

        if n % 100 == 0:
            print(f"  {n}/{len(dev)}")

    rows = [
        Row(
            arm=arm,
            metrics=aggregate(per_query[arm]),
            p50_ms=percentile(latencies[arm], 0.5),
            p95_ms=percentile(latencies[arm], 0.95),
            preamble_rank1=preamble[arm] / len(dev),
            right_doc_rank1=right_doc[arm] / len(dev),
            mean_docs_in_top10=float(np.mean(doc_spread[arm])),
        )
        for arm in arms
    ]

    # --- results ------------------------------------------------------------
    print(f"\n{'=' * 92}")
    print(f"PHASE 5c — QUERY PREFIX, {STRATEGY} @ {SIZE}, dense, {len(dev)} queries")
    print("=" * 92)
    print(f"\n  {'arm':<10}" + "".join(f"{m:>11}" for m in METRICS)
          + f"{'preamble':>10}{'p50 ms':>9}")
    print("  " + "-" * 85)
    for row in rows:
        cells = "".join(f"{row.metrics[f'{m}@10']:>11.4f}" for m in METRICS)
        print(f"  {row.arm:<10}{cells}{row.preamble_rank1:>9.1%}{row.p50_ms:>9.1f}")

    # --- document selection, measured separately -----------------------------
    print(f"\n{'=' * 92}")
    print("DOCUMENT SELECTION vs PASSAGE RETRIEVAL")
    print("=" * 92)
    print(f"\n  {'arm':<10}{'rank1 in right doc':>21}{'distinct docs in top10':>25}")
    print("  " + "-" * 56)
    for row in rows:
        print(f"  {row.arm:<10}{row.right_doc_rank1:>20.1%}"
              f"{row.mean_docs_in_top10:>25.2f}")

    print("\n  'rank1 in right doc' is whether the top result came from a document")
    print("  containing a gold span — document selection, scored on its own.")
    print("  'distinct docs in top10' shows how scattered retrieval was: 1.00 means")
    print("  all ten chunks came from one document.")

    if oracle_empty:
        print(f"\n  note: {oracle_empty} queries returned nothing in the oracle arm")
        print("  because their gold document produced no chunks in this configuration.")
        print("  They score zero there, so the oracle number is a slight understatement.")

    # --- verdict -------------------------------------------------------------
    full = rows[0]
    stripped = rows[1]
    oracle = rows[2]
    delta = stripped.metrics["recall@10"] - full.metrics["recall@10"]
    oracle_gap = oracle.metrics["recall@10"] - full.metrics["recall@10"]

    print(f"\n{'=' * 92}")
    print("VERDICT")
    print("=" * 92)
    print(f"\n  stripping the prefix: recall@10 {full.metrics['recall@10']:.4f} -> "
          f"{stripped.metrics['recall@10']:.4f}  ({delta:+.4f})")
    print(f"  preamble rate       : {full.preamble_rank1:.1%} -> "
          f"{stripped.preamble_rank1:.1%}")
    print(f"  right document      : {full.right_doc_rank1:.1%} -> "
          f"{stripped.right_doc_rank1:.1%}")

    if delta > 0.01:
        print("\n  -> The prefix was hurting more than helping. Its party names pulled")
        print("     retrieval toward document preambles, and removing them recovered")
        print("     more than the lost document signal cost.")
    elif delta < -0.01:
        print("\n  -> The prefix was doing necessary work. Without the party names,")
        print("     retrieval cannot tell which of 714 contracts is meant, and that")
        print("     costs more than the preamble bias did.")
    else:
        print("\n  -> Roughly flat. The two effects cancel: the prefix causes the")
        print("     preamble bias and also supplies the only document signal, so")
        print("     removing it trades one failure for another. The real fix is")
        print("     architectural, not a query edit.")

    print(f"\n  ORACLE upper bound  : recall@10 = {oracle.metrics['recall@10']:.4f} "
          f"({oracle_gap:+.4f} over full)")
    print(f"  precision           : {full.metrics['precision@10']:.4f} -> "
          f"{oracle.metrics['precision@10']:.4f}")
    print("\n  The oracle arm restricts search to the document the gold span is in,")
    print("  using the answer key. No real system could do this, which is exactly")
    print("  why it is an upper bound rather than a result: it is the performance")
    print("  currently lost to searching 714 documents at once when the right one")
    print("  was already identifiable from the query.")

    if examples:
        print(f"\n{'=' * 92}")
        print("QUERIES WHERE STRIPPING CHANGED THE OUTCOME")
        print("=" * 92)
        gained = sorted([e for e in examples if e["delta"] > 0],
                        key=lambda e: -e["delta"])[:N_EXAMPLES]
        lost = sorted([e for e in examples if e["delta"] < 0],
                      key=lambda e: e["delta"])[:N_EXAMPLES]

        for title, group in (
            ("STRIPPING HELPED", gained),
            ("STRIPPING HURT", lost),
        ):
            if not group:
                continue
            print(f"\n  {title}")
            for e in group:
                print(f"\n    [{e['query_id']}]  {e['full_recall']:.2f} -> "
                      f"{e['stripped_recall']:.2f}")
                print(f"      stripped Q : {snippet(e['stripped_query'])}")
                print(f"      gold doc   : {e['gold_doc'].split('/')[-1][:60]}")
                print(f"      full top1  : {str(e['full_top1_doc']).split('/')[-1][:60]}")
                print(f"      strip top1 : {str(e['stripped_top1_doc']).split('/')[-1][:60]}")
                print(f"      docs in top10: {e['full_docs_in_top10']} full, "
                      f"{e['stripped_docs_in_top10']} stripped")

    # --- gate ----------------------------------------------------------------
    checks = {
        "the semicolon template holds for every query": len(with_semicolon) == len(dev),
        "all three arms scored every query": all(
            len(per_query[a]) == len(dev) for a in arms
        ),
        "oracle beats full (document filtering must help)": oracle_gap > 0,
        "oracle top1 is always in the right document": oracle.right_doc_rank1 > 0.99,
    }

    print(f"\n{'=' * 92}")
    print("--- automated checks ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You can state what the prefix experiment proved about the benchmark")
    print("  [ ] You can say what you would build if this were production")

    manifest_path = write_manifest(
        settings,
        {
            "phase": "5c",
            "experiment": "query_prefix_and_oracle_filter",
            "strategy": STRATEGY,
            "size": SIZE,
            "dev_subset_size": len(dev),
            "template_holds": len(with_semicolon) == len(dev),
            "prefix_share_of_query": round(1 - kept, 4),
            "stripped_delta_recall10": round(delta, 6),
            "oracle_gap_recall10": round(oracle_gap, 6),
            "automated_checks": checks,
            "rows": [asdict(r) for r in rows],
        },
    )

    (manifest_path.parent / "prefix_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
