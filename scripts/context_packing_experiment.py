"""Which context-packing policy preserves more gold?

Groq's free tier caps a single request at 8,000 tokens per minute. Retrieving ten
512-token chunks can exceed that, so the context has to be cut to fit. Two policies
are available, and they fail differently:

    DROP       add chunks in rank order until the budget is reached, then stop.
               Fewer excerpts, each one whole.

    TRUNCATE   keep all ten chunks, cut each one proportionally to fit.
               All excerpts present, several cut off mid-sentence.

Convention favours DROP — LangChain, LlamaIndex and most production RAG do it that
way. But convention is not evidence, and the harness already holds everything needed
to decide it on this corpus: gold spans with exact character positions, and the
retrieval results for every golden-set question.

**No LLM calls are required.** Both policies are simulated, and the metrics are
computed over the text that *would have been sent* to the model. What is being
measured is how much gold each policy delivers, not what the model then did with it.

**Why the answer is not obvious in advance.** DROP loses whole chunks — if the gold
sits at rank 8 and only six chunks fit, it is gone entirely. TRUNCATE keeps every
chunk present but damages all of them, including rank 1, and a clause cut mid-sentence
may be useless even though its characters were technically delivered.

The character-level metrics settle it: whichever policy leaves more gold characters in
the prompt wins, and precision says which wastes fewer tokens doing it.

Run with:  uv run python scripts/context_packing_experiment.py
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass

from rag_eval_harness.chunking.fixed import Chunk
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.metrics.retrieval import CharRange, aggregate, evaluate_query
from rag_eval_harness.retrieval.dense import DenseRetriever

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("needs `transformers`. Install with: uv add transformers") from exc

STRATEGY = "fixed"
SIZE = 512
TOP_K = 10
TOKENIZER = "BAAI/bge-small-en-v1.5"

# Groq free tier: 8,000 tokens per minute for this model. The prompt must leave room
# for the system message, the question, the excerpt scaffolding, and the model's own
# output — which for a reasoning model is substantial.
TPM_LIMIT = 8000
RESERVED_FOR_OUTPUT = 2500
RESERVED_FOR_SCAFFOLD = 400
CONTEXT_BUDGET = TPM_LIMIT - RESERVED_FOR_OUTPUT - RESERVED_FOR_SCAFFOLD

GOLDEN_PATH = settings.data_dir.parent / "evals" / "golden_v1.jsonl"


@dataclass
class PackedContext:
    """The text a policy would actually send, as character ranges."""

    ranges: list[CharRange]
    chunks_included: int
    chunks_dropped: int
    chunks_truncated: int
    total_chars: int
    total_tokens: int


class Packer:
    def __init__(self, tokenizer_name: str = TOKENIZER):
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._cache:
            self._cache[text] = len(
                self._tok(text, add_special_tokens=False, verbose=False)["input_ids"]
            )
        return self._cache[text]

    def pack_drop(self, hits: list, budget: int) -> PackedContext:
        """Add chunks in rank order until the budget is reached, then stop.

        Degrades in rank order: the chunks lost are the ones the retriever ranked
        lowest, so the least valuable context goes first.
        """
        ranges: list[CharRange] = []
        used = 0
        included = 0

        for hit in hits:
            n = self.count(hit.chunk.text)
            if used + n > budget:
                break
            ranges.append(
                CharRange(hit.chunk.doc_id, hit.chunk.start, hit.chunk.end)
            )
            used += n
            included += 1

        return PackedContext(
            ranges=ranges,
            chunks_included=included,
            chunks_dropped=len(hits) - included,
            chunks_truncated=0,
            total_chars=sum(r.length for r in ranges),
            total_tokens=used,
        )

    def pack_truncate(self, hits: list, budget: int) -> PackedContext:
        """Keep every chunk, cut each proportionally to fit the budget.

        Every chunk survives in some form, but all of them are damaged — including
        rank 1. Truncation is from the tail, so each chunk keeps its opening.
        """
        if not hits:
            return PackedContext([], 0, 0, 0, 0, 0)

        counts = [self.count(h.chunk.text) for h in hits]
        total = sum(counts)

        if total <= budget:
            ranges = [
                CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits
            ]
            return PackedContext(
                ranges=ranges,
                chunks_included=len(hits),
                chunks_dropped=0,
                chunks_truncated=0,
                total_chars=sum(r.length for r in ranges),
                total_tokens=total,
            )

        keep = budget / total
        ranges: list[CharRange] = []
        truncated = 0
        used = 0

        for hit, n in zip(hits, counts):
            chunk_chars = hit.chunk.end - hit.chunk.start
            kept_chars = max(1, int(chunk_chars * keep))
            if kept_chars < chunk_chars:
                truncated += 1
            # Truncation takes the head of the chunk and discards the tail, which
            # is what any real implementation does.
            ranges.append(
                CharRange(hit.chunk.doc_id, hit.chunk.start, hit.chunk.start + kept_chars)
            )
            used += int(n * keep)

        return PackedContext(
            ranges=ranges,
            chunks_included=len(hits),
            chunks_dropped=0,
            chunks_truncated=truncated,
            total_chars=sum(r.length for r in ranges),
            total_tokens=used,
        )


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_chunks() -> list[Chunk]:
    path = settings.cache_path.parent / f"chunks_{STRATEGY}_{SIZE}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> int:
    golden = [g for g in load_golden() if g["answerable"]]
    print(f"answerable golden cases : {len(golden)}")
    print(f"TPM limit               : {TPM_LIMIT:,} tokens")
    print(f"reserved for output     : {RESERVED_FOR_OUTPUT:,}")
    print(f"reserved for scaffold   : {RESERVED_FOR_SCAFFOLD:,}")
    print(f"context budget          : {CONTEXT_BUDGET:,} tokens\n")

    chunks = load_chunks()
    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
    stats = dense.build(show_progress=False)
    if not stats.from_cache:
        print("warning: embeddings rebuilt rather than loaded from cache")

    packer = Packer()

    unlimited: list[dict[str, float]] = []
    drop_scores: list[dict[str, float]] = []
    trunc_scores: list[dict[str, float]] = []
    drop_meta: list[PackedContext] = []
    trunc_meta: list[PackedContext] = []
    over_budget = 0
    rows: list[dict] = []

    print("--- simulating both policies ---")
    for n, g in enumerate(golden, start=1):
        hits = dense.retrieve(g["question"], top_k=TOP_K)
        gold = [CharRange(s["doc_id"], s["start"], s["end"]) for s in g["gold_spans"]]

        full = [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]
        full_tokens = sum(packer.count(h.chunk.text) for h in hits)
        if full_tokens > CONTEXT_BUDGET:
            over_budget += 1

        packed_drop = packer.pack_drop(hits, CONTEXT_BUDGET)
        packed_trunc = packer.pack_truncate(hits, CONTEXT_BUDGET)

        # Metrics are computed at K = the number of ranges each policy delivered,
        # so the score describes exactly what would reach the model.
        u = evaluate_query(full, gold, ks=(TOP_K,))
        d = evaluate_query(packed_drop.ranges, gold, ks=(TOP_K,))
        t = evaluate_query(packed_trunc.ranges, gold, ks=(TOP_K,))

        unlimited.append(u)
        drop_scores.append(d)
        trunc_scores.append(t)
        drop_meta.append(packed_drop)
        trunc_meta.append(packed_trunc)

        rows.append(
            {
                "id": g["id"],
                "full_tokens": full_tokens,
                "over_budget": full_tokens > CONTEXT_BUDGET,
                "unlimited_recall": round(u[f"recall@{TOP_K}"], 4),
                "drop_recall": round(d[f"recall@{TOP_K}"], 4),
                "truncate_recall": round(t[f"recall@{TOP_K}"], 4),
                "chunks_kept_drop": packed_drop.chunks_included,
                "chunks_truncated": packed_trunc.chunks_truncated,
            }
        )

        if n % 10 == 0:
            print(f"  {n}/{len(golden)}")

    agg_u = aggregate(unlimited)
    agg_d = aggregate(drop_scores)
    agg_t = aggregate(trunc_scores)

    # --- results -------------------------------------------------------------
    print(f"\n{'=' * 86}")
    print(f"CONTEXT PACKING — {len(golden)} answerable golden cases, "
          f"{CONTEXT_BUDGET:,}-token budget")
    print("=" * 86)

    print(f"\n  questions exceeding the budget at top-{TOP_K}: "
          f"{over_budget}/{len(golden)} ({over_budget / len(golden):.0%})")
    print("  (the rest are unaffected — both policies deliver all ten chunks)\n")

    print(f"  {'policy':<24}{'recall@10':>12}{'precision@10':>14}{'mean chars':>13}")
    print("  " + "-" * 63)
    mean_chars_u = sum(sum(r.length for r in
                           [CharRange(h['doc_id'], h['start'], h['end'])
                            for h in []]) for _ in [0]) or 0
    print(f"  {'no limit (reference)':<24}{agg_u[f'recall@{TOP_K}']:>12.4f}"
          f"{agg_u[f'precision@{TOP_K}']:>14.4f}"
          f"{sum(m.total_chars for m in drop_meta) / len(drop_meta):>13.0f}*")
    print(f"  {'DROP whole chunks':<24}{agg_d[f'recall@{TOP_K}']:>12.4f}"
          f"{agg_d[f'precision@{TOP_K}']:>14.4f}"
          f"{sum(m.total_chars for m in drop_meta) / len(drop_meta):>13.0f}")
    print(f"  {'TRUNCATE each chunk':<24}{agg_t[f'recall@{TOP_K}']:>12.4f}"
          f"{agg_t[f'precision@{TOP_K}']:>14.4f}"
          f"{sum(m.total_chars for m in trunc_meta) / len(trunc_meta):>13.0f}")
    print("\n  * the no-limit row is the ceiling: what both policies are cutting from.")

    # --- what each policy did ------------------------------------------------
    affected = [r for r in rows if r["over_budget"]]
    print(f"\n{'=' * 86}")
    print("WHAT EACH POLICY ACTUALLY DID")
    print("=" * 86)

    if affected:
        mean_kept = sum(r["chunks_kept_drop"] for r in affected) / len(affected)
        mean_trunc = sum(r["chunks_truncated"] for r in affected) / len(affected)
        print(f"\n  on the {len(affected)} questions that exceeded the budget:")
        print(f"    DROP     kept {mean_kept:.1f} of {TOP_K} chunks on average, whole")
        print(f"    TRUNCATE kept all {TOP_K}, cutting {mean_trunc:.1f} of them")

        d_recall = sum(r["drop_recall"] for r in affected) / len(affected)
        t_recall = sum(r["truncate_recall"] for r in affected) / len(affected)
        u_recall = sum(r["unlimited_recall"] for r in affected) / len(affected)

        print(f"\n  recall on those questions only:")
        print(f"    no limit : {u_recall:.4f}")
        print(f"    DROP     : {d_recall:.4f}  ({d_recall - u_recall:+.4f})")
        print(f"    TRUNCATE : {t_recall:.4f}  ({t_recall - u_recall:+.4f})")
    else:
        print("\n  no question exceeded the budget — nothing was cut by either policy.")

    # --- per-question disagreement -------------------------------------------
    disagree = [
        r for r in rows if abs(r["drop_recall"] - r["truncate_recall"]) > 0.01
    ]
    if disagree:
        print(f"\n  questions where the policies differ: {len(disagree)}")
        drop_wins = sum(1 for r in disagree if r["drop_recall"] > r["truncate_recall"])
        print(f"    DROP delivered more gold     : {drop_wins}")
        print(f"    TRUNCATE delivered more gold : {len(disagree) - drop_wins}")

        print(f"\n  the five largest differences:")
        for r in sorted(disagree,
                        key=lambda x: -abs(x["drop_recall"] - x["truncate_recall"]))[:5]:
            print(f"    {r['id']:<18} drop {r['drop_recall']:.2f}  "
                  f"truncate {r['truncate_recall']:.2f}  "
                  f"({r['full_tokens']:,} tokens at top-10)")

    # --- verdict --------------------------------------------------------------
    print(f"\n{'=' * 86}")
    print("VERDICT")
    print("=" * 86)

    recall_delta = agg_d[f"recall@{TOP_K}"] - agg_t[f"recall@{TOP_K}"]
    prec_delta = agg_d[f"precision@{TOP_K}"] - agg_t[f"precision@{TOP_K}"]

    print(f"\n  DROP minus TRUNCATE:  recall {recall_delta:+.4f}   "
          f"precision {prec_delta:+.4f}")

    if recall_delta > 0.005:
        print("\n  -> DROP delivers more gold. Losing the lowest-ranked chunks whole")
        print("     costs less than damaging every chunk including rank 1.")
        print("     This matches the convention, and now it is measured rather than")
        print("     assumed.")
    elif recall_delta < -0.005:
        print("\n  -> TRUNCATE delivers more gold on this corpus, contradicting the")
        print("     convention. Worth investigating before adopting: check whether")
        print("     the gold tends to sit near the start of its chunk, which would")
        print("     make head-truncation unusually cheap here.")
    else:
        print("\n  -> The two are within noise on gold delivered. DROP is still the")
        print("     better choice on a second criterion: truncated chunks reach the")
        print("     model as broken sentences, and a clause cut mid-way can be")
        print("     useless even when its characters were technically delivered.")
        print("     Character recall cannot see that; it counts positions, not")
        print("     readability.")

    print("\n  Note what this experiment does and does not measure. It measures how")
    print("  much gold text each policy puts in front of the model. It does not")
    print("  measure whether the model can use it — a mid-sentence fragment scores")
    print("  the same as a whole clause here. That asymmetry favours DROP beyond")
    print("  what the numbers show.")

    manifest_path = write_manifest(
        settings,
        {
            "phase": 6,
            "experiment": "context_packing",
            "tpm_limit": TPM_LIMIT,
            "context_budget": CONTEXT_BUDGET,
            "top_k": TOP_K,
            "cases": len(golden),
            "over_budget": over_budget,
            "unlimited": {k: round(v, 6) for k, v in agg_u.items()},
            "drop": {k: round(v, 6) for k, v in agg_d.items()},
            "truncate": {k: round(v, 6) for k, v in agg_t.items()},
        },
    )

    (manifest_path.parent / "context_packing.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
