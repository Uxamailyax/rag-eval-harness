"""Phase 7, step 3: score all fifty answers with the calibrated judge.

Calibration established that the judge agrees with a human at kappa 0.87 on thirty
cases, with zero false positives — it never passed an answer the human failed. That is
what makes the numbers below worth reporting: the judge has been measured, so its
verdicts on the twenty cases nobody labelled carry the same weight as the thirty that
were.

**This is the step the whole phase exists to license.** Running a judge over fifty
answers takes an afternoon. Knowing whether to believe it takes the calibration. Almost
nobody does the second part, and a faithfulness score reported without it is an
unvalidated opinion from a model.

**What is reported, and why each piece is needed**

    faithfulness rate       the headline: what fraction of answers are grounded
    by stratum              answerable vs unanswerable, grounded vs refused
    against retrieval       does unfaithfulness correlate with retrieval failure?
    failure types           unsupported, contradicted, unsupported absence

That last breakdown matters because the three types have different fixes. Unsupported
claims point at the generator; unsupported absences point at the prompt; contradictions
point at the model reading carelessly.

**The calibration interval travels with the number.** Reporting "faithfulness 0.74"
alone implies a precision the measurement does not have. Reporting it alongside
kappa 0.87 (95% CI 0.67-1.00) says how much to trust it.

Run with:  uv run python scripts/phase7_score_all.py
           uv run python scripts/phase7_score_all.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.judge import FaithfulnessJudge, JUDGE_PROMPT_VERSION

EVALS = settings.data_dir.parent / "evals"
ANSWERS_PATH = EVALS / "answers_v2.jsonl"
SCORES_PATH = EVALS / "faithfulness_v2.jsonl"

# From scripts/phase7_calibrate.py. Carried here so the headline number never
# appears without the evidence that licenses it.
CALIBRATION = {
    "kappa": 0.8667,
    "ci_95": [0.6667, 1.0000],
    "raw_agreement": 0.9333,
    "cases": 30,
    "false_positives": 0,
    "false_negatives": 2,
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_done() -> dict[str, dict]:
    if not SCORES_PATH.exists():
        return {}
    return {r["id"]: r for r in load_jsonl(SCORES_PATH)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not ANSWERS_PATH.exists():
        print(f"missing {ANSWERS_PATH}")
        return 1

    answers = load_jsonl(ANSWERS_PATH)
    done = load_done()
    remaining = [a for a in answers if a["id"] not in done]

    print(f"answers    : {len(answers)}")
    print(f"scored     : {len(done)}")
    print(f"remaining  : {len(remaining)}\n")

    if remaining:
        source = LegalBenchRAG(settings.data_dir)
        judge = FaithfulnessJudge()
        print(f"judge      : {judge.model}")
        print(f"calibrated : kappa {CALIBRATION['kappa']:.2f} "
              f"(95% CI {CALIBRATION['ci_95'][0]:.2f}-{CALIBRATION['ci_95'][1]:.2f})\n")

        target = remaining[: args.limit] if args.limit else remaining
        print(f"--- scoring {len(target)} answers ---\n")

        for n, a in enumerate(target, start=1):
            # Only the chunks actually sent to the generator are shown to the judge.
            # Packing dropped some; judging against context the answerer never saw
            # would measure a different thing.
            sent = [h for h in a["retrieved"] if h.get("sent", True)]
            context = judge.build_context(sent, source)

            verdict = judge.judge(a["question"], a["answer"], context)

            record = {
                "id": a["id"],
                "answerable": a["answerable"],
                "subset": a["subset"],
                "faithful": 1 if verdict.faithful else 0,
                "failure_type": verdict.failure_type,
                "reason": verdict.reason,
                "precheck": verdict.precheck,
                "retrieval_recall_10": a["retrieval_recall_10"],
                "answer_chars": len(a["answer"]),
                "chunks_sent": a.get("chunks_sent"),
                "prompt_tokens": verdict.prompt_tokens,
                "completion_tokens": verdict.completion_tokens,
                "judge_model": judge.model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
            }
            with open(SCORES_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            mark = "F" if verdict.faithful else "U"
            tag = "precheck" if verdict.precheck else (
                "cache" if verdict.cached else f"{verdict.prompt_tokens or 0} tok"
            )
            print(f"  [{n}/{len(target)}] {a['id']:<18} {mark}  {tag:>9}  "
                  f"{verdict.reason[:48]}")

        print(f"\n  {judge.stats()}")
        done = load_done()

    if len(done) < len(answers):
        left = len(answers) - len(done)
        print(f"\n{left} answers still unscored. Run again to continue.")
        return 0

    rows = [done[a["id"]] for a in answers]

    # --- headline ------------------------------------------------------------
    faithful = sum(r["faithful"] for r in rows)
    rate = faithful / len(rows)

    print(f"\n{'=' * 82}")
    print("FAITHFULNESS — ALL 50 ANSWERS")
    print("=" * 82)
    print(f"\n  faithful   : {faithful}/{len(rows)}  ({rate:.1%})")
    print(f"  unfaithful : {len(rows) - faithful}/{len(rows)}")
    print(f"\n  judge calibrated at kappa {CALIBRATION['kappa']:.2f} "
          f"(95% CI {CALIBRATION['ci_95'][0]:.2f}-{CALIBRATION['ci_95'][1]:.2f}) "
          f"against {CALIBRATION['cases']} human labels,")
    print(f"  with {CALIBRATION['false_positives']} false positives — it never passed "
          f"an answer the human failed.")

    # --- by answerability ----------------------------------------------------
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    print(f"\n{'=' * 82}")
    print("BY QUESTION TYPE")
    print("=" * 82)
    print(f"\n  {'':<34}{'cases':>8}{'faithful':>11}{'rate':>9}")
    print("  " + "-" * 62)
    for label, group in (
        ("answerable", answerable),
        ("unanswerable (wrong document)", unanswerable),
    ):
        if group:
            f = sum(r["faithful"] for r in group)
            print(f"  {label:<34}{len(group):>8}{f:>11}{f / len(group):>9.0%}")

    # --- against retrieval ---------------------------------------------------
    zero = [r for r in answerable if r["retrieval_recall_10"] == 0]
    some = [r for r in answerable if r["retrieval_recall_10"] > 0]

    print(f"\n{'=' * 82}")
    print("FAITHFULNESS AGAINST RETRIEVAL QUALITY")
    print("=" * 82)
    print(f"\n  {'':<34}{'cases':>8}{'faithful':>11}{'rate':>9}")
    print("  " + "-" * 62)
    for label, group in (
        ("retrieval found gold", some),
        ("retrieval found no gold", zero),
    ):
        if group:
            f = sum(r["faithful"] for r in group)
            print(f"  {label:<34}{len(group):>8}{f:>11}{f / len(group):>9.0%}")

    if zero and some:
        z_rate = sum(r["faithful"] for r in zero) / len(zero)
        s_rate = sum(r["faithful"] for r in some) / len(some)
        print()
        if z_rate < s_rate - 0.1:
            print("  -> Unfaithfulness concentrates where retrieval failed. The")
            print("     generator asserts more when it has less to work from, which")
            print("     is the condition that produces hallucination.")
        elif abs(z_rate - s_rate) <= 0.1:
            print("  -> Faithfulness is roughly independent of retrieval quality. The")
            print("     generator neither improves nor degrades with better context —")
            print("     it grounds what it says either way.")

    # --- failure types -------------------------------------------------------
    failures = [r for r in rows if not r["faithful"]]
    if failures:
        types = Counter(r["failure_type"] for r in failures)
        print(f"\n{'=' * 82}")
        print("HOW ANSWERS FAILED")
        print("=" * 82)
        print()
        labels = {
            "unsupported": "asserts something the excerpts do not contain",
            "contradicted": "the excerpts say the opposite",
            "unsupported_absence": "claims an absence the excerpts cannot establish",
            "unknown": "type not parsed from the judge's response",
        }
        for t, count in types.most_common():
            print(f"  {count:>3}  {t:<22} {labels.get(t, '')}")

        print("\n  The three types have different fixes: unsupported claims point at")
        print("  the generator, unsupported absences at the prompt, contradictions at")
        print("  the model reading carelessly.")

    # --- the unfaithful answers ----------------------------------------------
    if failures:
        print(f"\n{'=' * 82}")
        print(f"THE {len(failures)} UNFAITHFUL ANSWERS")
        print("=" * 82)
        for r in failures:
            mark = "?" if not r["answerable"] else " "
            print(f"\n  [{r['id']}]{mark} {r['failure_type']}  "
                  f"recall {r['retrieval_recall_10']:.2f}")
            print(f"    {r['reason'][:150]}")

    # --- precheck saving -----------------------------------------------------
    prechecked = sum(1 for r in rows if r.get("precheck"))
    pt = sum(r.get("prompt_tokens") or 0 for r in rows)
    ct = sum(r.get("completion_tokens") or 0 for r in rows)
    judged = len(rows) - prechecked
    # Groq list price for the judge model is not published; gpt-oss-120b rates are
    # used as a stand-in so the figure has a scale rather than being free-tier zero.
    cost = (pt / 1e6) * 0.15 + (ct / 1e6) * 0.60

    print(f"\n{'=' * 82}")
    print("JUDGE COST")
    print("=" * 82)
    print(f"\n  resolved by precheck : {prechecked}/{len(rows)} "
          f"({prechecked / len(rows):.0%})")
    print(f"  judged by the model  : {judged}")
    print(f"  tokens               : {pt:,} in / {ct:,} out")
    print(f"  cost for 50          : ${cost:.4f}")
    print(f"  cost per answer      : ${cost / len(rows):.5f}")
    print(f"\n  Prechecks saved {prechecked} calls. Kept deliberately narrow — only a")
    print(f"  bare \"I don't know\" — because Phase 6 showed that matching on")
    print(f"  refusal-like phrasing swallows genuine negative assertions.")

    write_manifest(
        settings,
        {
            "phase": 7,
            "step": "score_all",
            "judge_model": rows[0]["judge_model"],
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "calibration": CALIBRATION,
            "answers": len(rows),
            "faithful": faithful,
            "faithfulness_rate": round(rate, 4),
            "by_answerable": {
                "answerable": {
                    "n": len(answerable),
                    "faithful": sum(r["faithful"] for r in answerable),
                },
                "unanswerable": {
                    "n": len(unanswerable),
                    "faithful": sum(r["faithful"] for r in unanswerable),
                },
            },
            "by_retrieval": {
                "gold_found": {
                    "n": len(some),
                    "faithful": sum(r["faithful"] for r in some),
                },
                "no_gold": {
                    "n": len(zero),
                    "faithful": sum(r["faithful"] for r in zero),
                },
            },
            "failure_types": dict(Counter(r["failure_type"] for r in failures)),
            "precheck_resolved": prechecked,
            "judge_tokens": {"prompt": pt, "completion": ct},
            "judge_cost_usd": round(cost, 4),
        },
    )

    print(f"\n  scores: {SCORES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
