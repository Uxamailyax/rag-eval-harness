"""When does the model refuse, and when does it answer anyway?

Phase 6 produced 50 answers. This classifies each one as a refusal or an attempted
answer, and cross-references that against two things the answer set already records:
whether retrieval found any gold, and whether the question was answerable at all.

**Why a proper refusal detector is needed.** A first pass matched on the literal
string "know", on the assumption that refusals use the prompt's exact wording. They
do not. `answerable:09` refused with "The excerpts do not contain any provision that
sets a monetary..." and `answerable:14` with "The FullStory Mutual NDA does not
contain any provision that allows..." — both genuine refusals, neither containing the
expected phrase.

That crude filter would have counted two correct refusals as hallucinations: a 4%
error injected into the headline number, pointing in the worst possible direction.
**When classifying free-text model output, never assume it uses your exact wording.**

The patterns below were written by reading the actual outputs, not by guessing at
them. Every classification is printed for inspection, because a regex over model
output is itself an unvalidated judgement.

**The two conditions this separates**

    retrieval failure     a real question about the right document, where retrieval
                          returned no gold. The chunks are visibly off-topic.

    wrong document        a real question pointed at a document the benchmark's
                          annotators recorded no answer of that type in. The chunks
                          look entirely appropriate — right document, right domain,
                          plausible clauses.

These look identical from the outside (no answer available) and produce opposite
behaviour, which is the Phase 6 finding.

Run with:  uv run python scripts/phase6_refusal_analysis.py
           uv run python scripts/phase6_refusal_analysis.py --show-all
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from rag_eval_harness.config import settings, write_manifest

ANSWERS_PATH = settings.data_dir.parent / "evals" / "answers_v1.jsonl"

# Written from the model's actual outputs. Ordered from the prompt's requested
# phrasing to the paraphrases it produced instead.
REFUSAL_PATTERNS = [
    (r"\bi don'?t know\b", "exact phrase requested by the prompt"),
    (r"\bexcerpts? (do|does) not (contain|include|provide|mention|address)\b",
     "negative finding about the excerpts"),
    (r"\bexcerpts? contain no\b", "negative finding about the excerpts"),
    (r"\b(does|do) not contain any provision\b", "negative finding about the document"),
    (r"\bthere is no (provision|clause|mention|reference|language)\b",
     "negative finding about the document"),
    (r"\b(no|not) (enough|sufficient) (information|context|detail)\b",
     "insufficient information"),
    (r"\bcannot (be )?(answer|determine|establish|tell)\b", "explicit inability"),
    (r"\bunable to (answer|determine|find)\b", "explicit inability"),
    (r"\bnot (stated|specified|addressed|mentioned) in the (excerpts?|provided)\b",
     "absence in the provided text"),
]

COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in REFUSAL_PATTERNS]


def classify(answer: str) -> tuple[bool, str]:
    """Is this a refusal, and on what basis?

    A refusal is any answer that declines to assert a fact about the contract. That
    includes the prompt's requested phrasing and every paraphrase the model produced
    instead.
    """
    for pattern, label in COMPILED:
        if pattern.search(answer):
            return True, label
    return False, "asserted an answer"


def truncate(text: str, n: int = 95) -> str:
    return " ".join(text.split())[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    if not ANSWERS_PATH.exists():
        print(f"no answers at {ANSWERS_PATH}\nRun scripts/phase6_generate.py first.")
        return 1

    with open(ANSWERS_PATH, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    for row in rows:
        row["refused"], row["refusal_basis"] = classify(row["answer"])

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    print(f"answers analysed : {len(rows)}")
    print(f"  answerable     : {len(answerable)}")
    print(f"  unanswerable   : {len(unanswerable)}\n")

    # --- how refusals were phrased -------------------------------------------
    refusals = [r for r in rows if r["refused"]]
    by_basis: dict[str, int] = {}
    for r in refusals:
        by_basis[r["refusal_basis"]] = by_basis.get(r["refusal_basis"], 0) + 1

    print("=" * 84)
    print("HOW THE MODEL PHRASED ITS REFUSALS")
    print("=" * 84)
    print(f"\n  {len(refusals)} of {len(rows)} answers were refusals\n")
    for basis, count in sorted(by_basis.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>3}  {basis}")

    exact = by_basis.get("exact phrase requested by the prompt", 0)
    paraphrased = len(refusals) - exact
    if paraphrased:
        print(f"\n  {exact} used the exact phrase the prompt asked for.")
        print(f"  {paraphrased} refused in their own words.")
        print(f"\n  A detector matching only the requested phrase would have counted")
        print(f"  those {paraphrased} as hallucinations — a {paraphrased / len(rows):.0%} "
              f"error in the headline number,")
        print(f"  inventing failures that did not occur.")

    # --- condition 1: retrieval failure --------------------------------------
    zero_recall = [r for r in answerable if r["retrieval_recall_10"] == 0]
    some_recall = [r for r in answerable if r["retrieval_recall_10"] > 0]

    zr_refused = sum(1 for r in zero_recall if r["refused"])
    sr_refused = sum(1 for r in some_recall if r["refused"])

    print(f"\n{'=' * 84}")
    print("CONDITION 1 — RETRIEVAL FAILURE (answerable question, no gold retrieved)")
    print("=" * 84)
    print(f"\n  {'':<28}{'count':>8}{'refused':>10}{'rate':>9}")
    print("  " + "-" * 55)
    if zero_recall:
        print(f"  {'retrieval found no gold':<28}{len(zero_recall):>8}"
              f"{zr_refused:>10}{zr_refused / len(zero_recall):>9.0%}")
    if some_recall:
        print(f"  {'retrieval found some gold':<28}{len(some_recall):>8}"
              f"{sr_refused:>10}{sr_refused / len(some_recall):>9.0%}")

    print("\n  Refusing when retrieval found nothing is correct behaviour. The chunks")
    print("  are visibly off-topic and the model declines rather than inventing.")

    # --- condition 2: wrong document -----------------------------------------
    un_refused = sum(1 for r in unanswerable if r["refused"])

    print(f"\n{'=' * 84}")
    print("CONDITION 2 — WRONG DOCUMENT (no answer of this type exists in it)")
    print("=" * 84)
    print(f"\n  cases            : {len(unanswerable)}")
    print(f"  refused          : {un_refused}")
    print(f"  answered anyway  : {len(unanswerable) - un_refused}")

    if len(unanswerable) - un_refused:
        print(f"\n  Every one of these was pointed at a document whose annotators")
        print(f"  labelled ~19 distinct question types, none of them this one.")
        print(f"  The model answered confidently, citing real contract language.")

    # --- the comparison that is the finding ----------------------------------
    print(f"\n{'=' * 84}")
    print("THE FINDING — two conditions that look identical, opposite behaviour")
    print("=" * 84)
    print(f"\n  {'condition':<42}{'refusal rate':>14}")
    print("  " + "-" * 56)
    if zero_recall:
        print(f"  {'retrieval failed (chunks off-topic)':<42}"
              f"{zr_refused / len(zero_recall):>14.0%}")
    if unanswerable:
        print(f"  {'wrong document (chunks look right)':<42}"
              f"{un_refused / len(unanswerable):>14.0%}")

    print("\n  Both mean no answer is available. The model handles one correctly and")
    print("  the other not at all.")
    print("\n  The mechanism: when retrieval fails, the chunks are visibly unrelated")
    print("  and the model notices. When the query names a real contract and")
    print("  retrieval returns that contract's chunks, the context looks entirely")
    print("  appropriate — same document, same legal domain, plausible clauses. So")
    print("  the model finds something adjacent and answers from it.")
    print("\n  This is not invention from nothing. It is over-application of real")
    print("  context, which is harder to detect precisely because every quote is")
    print("  genuine.")

    # --- the hallucinations, in full -----------------------------------------
    hallucinations = [r for r in unanswerable if not r["refused"]]
    if hallucinations:
        print(f"\n{'=' * 84}")
        print(f"THE {len(hallucinations)} HALLUCINATIONS — read these")
        print("=" * 84)
        for r in hallucinations:
            print(f"\n  [{r['id']}]  {r['chunks_sent']}/{r['chunks_retrieved']} chunks sent")
            print(f"    Q: {truncate(r['question'], 140)}")
            print(f"    A: {truncate(r['answer'], 260)}")

    # --- answered-anyway cases among answerable questions --------------------
    invented = [r for r in zero_recall if not r["refused"]]
    if invented:
        print(f"\n{'=' * 84}")
        print(f"ANSWERABLE QUESTIONS ANSWERED WITHOUT GOLD — {len(invented)}")
        print("=" * 84)
        print("\n  Retrieval found no gold, and the model asserted an answer anyway.")
        print("  Worth reading: some may be correct from context the span labels")
        print("  do not cover, which is a different thing from a hallucination.\n")
        for r in invented:
            print(f"  [{r['id']}]  recall 0.00")
            print(f"    Q: {truncate(r['question'], 130)}")
            print(f"    A: {truncate(r['answer'], 200)}\n")

    if args.show_all:
        print(f"\n{'=' * 84}")
        print("ALL 50, CLASSIFIED")
        print("=" * 84)
        for r in rows:
            mark = "REFUSED " if r["refused"] else "ANSWERED"
            kind = "?" if not r["answerable"] else " "
            print(f"\n  {mark}{kind} [{r['id']}]  recall "
                  f"{r['retrieval_recall_10']:.2f}  ({r['refusal_basis']})")
            print(f"    {truncate(r['answer'], 150)}")

    # --- cost, which the answer records already carry ------------------------
    prompt_toks = sum(r.get("prompt_tokens") or 0 for r in rows)
    completion_toks = sum(r.get("completion_tokens") or 0 for r in rows)
    # Groq published rates for openai/gpt-oss-120b.
    cost = (prompt_toks / 1e6) * 0.15 + (completion_toks / 1e6) * 0.60

    print(f"\n{'=' * 84}")
    print("COST PER QUERY")
    print("=" * 84)
    print(f"\n  total tokens    : {prompt_toks:,} in / {completion_toks:,} out")
    print(f"  mean per query  : {prompt_toks // len(rows):,} in / "
          f"{completion_toks // len(rows):,} out")
    print(f"  cost for 50     : ${cost:.4f}   (Groq list: $0.15/$0.60 per M)")
    print(f"  cost per query  : ${cost / len(rows):.5f}")
    print(f"  at 1,000/day    : ${cost / len(rows) * 1000:.2f}/day")
    print(f"  at 100,000/day  : ${cost / len(rows) * 100_000:.0f}/day")
    print("\n  Free tier in practice, but priced at Groq's published rates so the")
    print("  figure means something outside this project.")

    manifest_path = write_manifest(
        settings,
        {
            "phase": 6,
            "step": "refusal_analysis",
            "answers": len(rows),
            "refusals_total": len(refusals),
            "refusals_exact_phrase": exact,
            "refusals_paraphrased": paraphrased,
            "zero_recall_cases": len(zero_recall),
            "zero_recall_refused": zr_refused,
            "unanswerable_cases": len(unanswerable),
            "unanswerable_refused": un_refused,
            "hallucinations": len(hallucinations),
            "tokens": {"prompt": prompt_toks, "completion": completion_toks},
            "cost_usd_50_queries": round(cost, 4),
            "cost_usd_per_query": round(cost / len(rows), 5),
        },
    )

    (manifest_path.parent / "refusal_classification.json").write_text(
        json.dumps(
            [
                {
                    "id": r["id"],
                    "answerable": r["answerable"],
                    "recall": r["retrieval_recall_10"],
                    "refused": r["refused"],
                    "basis": r["refusal_basis"],
                    "answer": r["answer"],
                }
                for r in rows
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nresults: {manifest_path.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
