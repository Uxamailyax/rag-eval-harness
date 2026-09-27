"""Phase 7, step 2: calibrate the judge against human labels.

The judge grades the same thirty cases a human already labelled. Cohen's kappa
measures how well the two agree, corrected for the agreement that would happen by
chance.

**The thirty are not the point — the judge is.** Those answers are already known.
What is unknown is whether the judge can be trusted on the other twenty, and on every
answer the system produces afterwards that nobody will ever label. The thirty are a
test set for the grader:

    kappa 0.85   the judge agrees where the answer is known, so trust it where it is not
    kappa 0.20   the judge is close to guessing, and every faithfulness score is noise

Same pattern as Phase 2, where the metrics were proved on toy data with known answers
before being trusted on real data.

**Why the confusion matrix matters more than the kappa.** A single number says how much
the two agree; it does not say *how* they disagree. A judge that calls everything
faithful and a judge that calls everything unfaithful can produce similar kappa and
need opposite fixes. The 2x2 shows which.

**Why a bootstrap interval.** Thirty cases is a small sample, and kappa on small
samples is unstable. Reporting 0.72 alone implies a precision the data does not
support; reporting 0.72 with a 95% interval of 0.48 to 0.91 is honest about it.

**Bias checks.** Verbosity bias is measured by correlating answer length with the
judge's verdict — a judge that simply prefers longer answers would show it there.
Self-preference is controlled by construction: the generator is
`openai/gpt-oss-120b`, the judge is `qwen/qwen3.6-27b`, different families with
different training data.

**A low kappa is a finding, not a failure.** Reported honestly with the confusion
matrix and an investigation of where the disagreements fall, it says more than a high
kappa left unexamined.

Run with:  uv run python scripts/phase7_calibrate.py
           uv run python scripts/phase7_calibrate.py --limit 5    (small batch)
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG
from rag_eval_harness.judge import FaithfulnessJudge, JUDGE_PROMPT_VERSION

EVALS = settings.data_dir.parent / "evals"
LABELSET_PATH = EVALS / "labelset_v1.jsonl"
HUMAN_PATH = EVALS / "human_labels_v1.jsonl"
JUDGE_PATH = EVALS / "judge_labels_v1.jsonl"

BOOTSTRAP_N = 10_000


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Agreement between two raters, corrected for chance.

        kappa = (observed - expected) / (1 - expected)

    Returns 1.0 when both raters are perfectly consistent and expected agreement is
    total — the degenerate case where every label is identical. That is the kappa
    paradox: agreement of 100% on a one-sided distribution carries no information,
    because a rater that always says "yes" would score the same.
    """
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    a_yes = sum(a) / n
    b_yes = sum(b) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def bootstrap_kappa(a: list[int], b: list[int], n_samples: int, seed: int) -> tuple:
    """Percentile confidence interval for kappa by resampling cases with replacement."""
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx]))
    values.sort()
    return (
        values[int(0.025 * n_samples)],
        values[int(0.975 * n_samples)],
    )


def interpret(k: float) -> str:
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight — close to guessing"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial — usable"
    return "almost perfect"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_judge_done() -> dict[str, dict]:
    if not JUDGE_PATH.exists():
        return {}
    return {r["case_id"]: r for r in load_jsonl(JUDGE_PATH)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-prechecks", action="store_true")
    args = parser.parse_args()

    if not HUMAN_PATH.exists():
        print(f"missing {HUMAN_PATH}")
        print("Label the cases first, then rename the _BLANK file.")
        return 1

    cases = {c["case_id"]: c for c in load_jsonl(LABELSET_PATH)}
    human = {h["case_id"]: h for h in load_jsonl(HUMAN_PATH)}
    done = load_judge_done()

    unlabelled = [k for k, v in human.items() if v.get("faithful") is None]
    if unlabelled:
        print(f"{len(unlabelled)} cases still unlabelled: {unlabelled[:5]}")
        return 1

    remaining = [cid for cid in cases if cid not in done]
    print(f"cases        : {len(cases)}")
    print(f"human labels : {len(human)}")
    print(f"judge done   : {len(done)}")
    print(f"remaining    : {len(remaining)}\n")

    if remaining:
        source = LegalBenchRAG(settings.data_dir)
        judge = FaithfulnessJudge(use_prechecks=not args.no_prechecks)
        print(f"judge model  : {judge.model}")
        print(f"generator    : {settings.generator_model}")
        print(f"prechecks    : {'off' if args.no_prechecks else 'on'}\n")

        target = remaining[: args.limit] if args.limit else remaining
        print(f"--- judging {len(target)} cases ---\n")

        for n, case_id in enumerate(target, start=1):
            case = cases[case_id]
            context = judge.build_context(case["retrieved"], source)
            verdict = judge.judge(case["question"], case["answer"], context)

            record = {
                "case_id": case_id,
                "faithful": 1 if verdict.faithful else 0,
                "failure_type": verdict.failure_type,
                "reason": verdict.reason,
                "precheck": verdict.precheck,
                "cached": verdict.cached,
                "prompt_tokens": verdict.prompt_tokens,
                "completion_tokens": verdict.completion_tokens,
                "judge_model": judge.model,
                "prompt_version": JUDGE_PROMPT_VERSION,
            }
            with open(JUDGE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            tag = "precheck" if verdict.precheck else (
                "cache" if verdict.cached else f"{verdict.prompt_tokens or 0} tok"
            )
            mark = "F" if verdict.faithful else "U"
            print(f"  [{n}/{len(target)}] {case_id}  {mark}  {tag:>9}  "
                  f"{verdict.reason[:52]}")

        print(f"\n  {judge.stats()}")
        done = load_judge_done()

    if len(done) < len(cases):
        print(f"\n{len(cases) - len(done)} cases still unjudged. Run again to continue.")
        return 0

    # --- agreement -----------------------------------------------------------
    ids = sorted(cases)
    h = [human[i]["faithful"] for i in ids]
    j = [done[i]["faithful"] for i in ids]

    k = cohens_kappa(h, j)
    lo, hi = bootstrap_kappa(h, j, BOOTSTRAP_N, settings.random_seed)
    agreement = sum(1 for x, y in zip(h, j) if x == y) / len(ids)

    print(f"\n{'=' * 82}")
    print("JUDGE CALIBRATION")
    print("=" * 82)
    print(f"\n  cases                 : {len(ids)}")
    print(f"  raw agreement         : {agreement:.1%}")
    print(f"  Cohen's kappa         : {k:.4f}")
    print(f"  95% CI (bootstrap)    : {lo:.4f} to {hi:.4f}")
    print(f"  interpretation        : {interpret(k)}")

    # --- confusion matrix ----------------------------------------------------
    tp = sum(1 for x, y in zip(h, j) if x == 1 and y == 1)
    fn = sum(1 for x, y in zip(h, j) if x == 1 and y == 0)
    fp = sum(1 for x, y in zip(h, j) if x == 0 and y == 1)
    tn = sum(1 for x, y in zip(h, j) if x == 0 and y == 0)

    print(f"\n  confusion matrix (human vs judge)\n")
    print(f"                      judge: faithful   judge: unfaithful")
    print(f"    human: faithful   {tp:>15}   {fn:>17}")
    print(f"    human: unfaithful {fp:>15}   {tn:>17}")

    print(f"\n  human said faithful   : {sum(h)}/{len(h)}")
    print(f"  judge said faithful   : {sum(j)}/{len(j)}")

    if fp > fn * 2 and fp > 2:
        print("\n  -> The judge is too lenient: it passes answers the human failed.")
        print("     In production this under-reports hallucination, which is the")
        print("     more dangerous direction.")
    elif fn > fp * 2 and fn > 2:
        print("\n  -> The judge is too strict: it fails answers the human passed.")
        print("     Noisy, but it errs toward flagging rather than missing.")

    # --- where they disagree, by stratum -------------------------------------
    disagreements = [(i, human[i]["faithful"], done[i]["faithful"]) for i in ids
                     if human[i]["faithful"] != done[i]["faithful"]]

    if disagreements:
        print(f"\n{'=' * 82}")
        print(f"DISAGREEMENTS — {len(disagreements)}")
        print("=" * 82)
        by_stratum: dict[str, int] = {}
        for cid, _, _ in disagreements:
            s = cases[cid]["stratum"]
            by_stratum[s] = by_stratum.get(s, 0) + 1
        print(f"\n  by stratum: {by_stratum}\n")

        for cid, hv, jv in disagreements:
            case = cases[cid]
            print(f"  [{cid}] {case['stratum']:<10} human={hv} judge={jv}")
            print(f"    judge said: {done[cid]['reason'][:110]}")
            print(f"    answer    : {' '.join(case['answer'].split())[:110]}")
            print()

    # --- verbosity bias ------------------------------------------------------
    lengths = [len(cases[i]["answer"]) for i in ids]
    verdicts = [done[i]["faithful"] for i in ids]
    if len(set(verdicts)) > 1:
        corr = float(np.corrcoef(lengths, verdicts)[0, 1])
        print(f"{'=' * 82}")
        print("BIAS CHECKS")
        print("=" * 82)
        print(f"\n  verbosity: correlation between answer length and 'faithful'")
        print(f"    r = {corr:+.3f}")
        mean_f = np.mean([l for l, v in zip(lengths, verdicts) if v == 1])
        mean_u = np.mean([l for l, v in zip(lengths, verdicts) if v == 0])
        print(f"    mean length judged faithful   : {mean_f:.0f} chars")
        print(f"    mean length judged unfaithful : {mean_u:.0f} chars")
        if abs(corr) < 0.3:
            print("    -> no meaningful length preference")
        else:
            direction = "longer" if corr > 0 else "shorter"
            print(f"    -> the judge favours {direction} answers; verdicts may be")
            print(f"       tracking length rather than grounding")

        print(f"\n  self-preference: controlled by construction")
        print(f"    generator : {settings.generator_model}")
        print(f"    judge     : {done[ids[0]]['judge_model']}")
        print(f"    different model families, so the judge is not grading its own output")

    # --- cost ----------------------------------------------------------------
    pt = sum(done[i].get("prompt_tokens") or 0 for i in ids)
    ct = sum(done[i].get("completion_tokens") or 0 for i in ids)
    prechecked = sum(1 for i in ids if done[i].get("precheck"))

    print(f"\n{'=' * 82}")
    print("JUDGE COST")
    print("=" * 82)
    print(f"\n  cases judged by model : {len(ids) - prechecked}")
    print(f"  resolved by precheck  : {prechecked} ({prechecked / len(ids):.0%})")
    print(f"  tokens                : {pt:,} in / {ct:,} out")
    if len(ids) - prechecked:
        print(f"  mean per judged case  : {pt // max(1, len(ids) - prechecked):,} in")

    # --- the README line -----------------------------------------------------
    print(f"\n{'=' * 82}")
    print("FOR THE README")
    print("=" * 82)
    print(f"""
  LLM-as-judge calibrated against {len(ids)} stratified human labels,
  Cohen's kappa = {k:.2f} (95% CI {lo:.2f}-{hi:.2f}), raw agreement {agreement:.0%}.
  Judge is {done[ids[0]]['judge_model']}, a different model family from the
  generator ({settings.generator_model}), controlling for self-preference bias.
  Confusion matrix published; verbosity correlation r = {corr:+.2f}.
""")

    write_manifest(
        settings,
        {
            "phase": 7,
            "step": "calibrate",
            "judge_model": done[ids[0]]["judge_model"],
            "generator_model": settings.generator_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "cases": len(ids),
            "raw_agreement": round(agreement, 4),
            "cohens_kappa": round(k, 4),
            "kappa_ci_95": [round(lo, 4), round(hi, 4)],
            "confusion": {
                "human_faithful_judge_faithful": tp,
                "human_faithful_judge_unfaithful": fn,
                "human_unfaithful_judge_faithful": fp,
                "human_unfaithful_judge_unfaithful": tn,
            },
            "human_faithful_count": sum(h),
            "judge_faithful_count": sum(j),
            "disagreements": len(disagreements),
            "verbosity_correlation": round(corr, 4),
            "precheck_resolved": prechecked,
            "judge_tokens": {"prompt": pt, "completion": ct},
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
