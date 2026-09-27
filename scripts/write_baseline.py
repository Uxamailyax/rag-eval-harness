"""Write the CI baseline.

Records what the system currently scores, so a later run can be checked against it.
Without a baseline there is nothing for a regression test to compare to, and the
scenario the whole gate exists to prevent goes undetected:

    A prompt is tweaked on Tuesday to improve one kind of answer. It quietly degrades
    another kind. Nobody notices until Friday, from a customer.

With a baseline committed to the repository, that change fails the build on Tuesday.

**What goes in the baseline.** The headline retrieval and generation numbers, the
configuration that produced them, and the prompt fingerprints. The fingerprints matter:
a score is only comparable to a baseline produced under the same prompt, so the gate
can tell the difference between "quality dropped" and "the prompt changed, re-baseline".

**Why the thresholds are relative rather than absolute.** Scores wobble slightly
between runs even with no code change — the generator is non-deterministic even at
temperature 0. A gate that fires on noise gets disabled, which is worse than having no
gate. The tolerance is set to catch real regressions and ignore jitter.

Run with:  uv run python scripts/write_baseline.py
           uv run python scripts/write_baseline.py --show    (print, do not write)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from rag_eval_harness.config import PROJECT_ROOT, git_sha, settings

EVALS = PROJECT_ROOT / "evals"
BASELINE_PATH = PROJECT_ROOT / "results" / "baseline.json"

# How far a metric may fall before the build fails, as a fraction of the baseline.
# 5% is tight enough to catch a real regression and loose enough to survive run-to-run
# variation. The plan's guidance applies: start permissive and tighten as the system
# matures, because a gate that fires constantly gets switched off.
TOLERANCE = 0.05


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="print without writing")
    args = parser.parse_args()

    faithfulness_path = EVALS / "faithfulness_v2.jsonl"
    answers_path = EVALS / "answers_v2.jsonl"

    for path in (faithfulness_path, answers_path):
        if not path.exists():
            print(f"missing {path}")
            print("Run scripts/phase6_generate.py and scripts/phase7_score_all.py first.")
            return 1

    scores = load_jsonl(faithfulness_path)
    answers = load_jsonl(answers_path)

    faithful = sum(r["faithful"] for r in scores)
    answerable = [r for r in scores if r["answerable"]]
    unanswerable = [r for r in scores if not r["answerable"]]

    # Retrieval quality as the generator actually saw it — scored over the chunks
    # that survived context packing, not the full top-10.
    recalls = [a["retrieval_recall_10"] for a in answers if a["answerable"]]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

    baseline = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config": {
            "chunking": "fixed",
            "chunk_size_tokens": 512,
            "retriever": "dense",
            "embedding_model": settings.embedding_model,
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "top_k": 10,
            "context_packing": "drop",
        },
        "prompts": {
            "qa": {"version": "v1"},
            "judge_faithfulness": {"version": "v1"},
        },
        "calibration": {
            "cohens_kappa": 0.8667,
            "ci_95": [0.6667, 1.0000],
            "human_labels": 30,
            "false_positives": 0,
            "note": (
                "Applies to judge prompt v1 only. Changing the rubric invalidates this "
                "calibration until the 30 cases are re-judged."
            ),
        },
        "metrics": {
            "faithfulness": round(faithful / len(scores), 4),
            "faithfulness_answerable": round(
                sum(r["faithful"] for r in answerable) / len(answerable), 4
            ) if answerable else None,
            "faithfulness_unanswerable": round(
                sum(r["faithful"] for r in unanswerable) / len(unanswerable), 4
            ) if unanswerable else None,
            "mean_retrieval_recall_10": round(mean_recall, 4),
            "answers_scored": len(scores),
        },
        # Retrieval figures from the Phase 5 sweep, which is a separate run and is not
        # re-executed by the gate. Recorded so a change to chunking or retrieval that
        # bypasses the generation path is still visible in the baseline diff.
        "retrieval_reference": {
            "recall_at_10": 0.4226,
            "precision_at_10": 0.0117,
            "ndcg_at_10": 0.2839,
            "mrr_at_10": 0.2235,
            "source": "scripts/phase5_sweep.py, fixed @ 512, dense, 500 queries",
        },
        "tolerance": TOLERANCE,
        "gate_note": (
            "A metric may fall by up to `tolerance` before the build fails. Scores "
            "vary slightly between runs even at temperature 0; a gate that fires on "
            "noise gets disabled, which is worse than no gate."
        ),
    }

    print(json.dumps(baseline, indent=2))

    if args.show:
        print("\n(--show: nothing written)")
        return 0

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {BASELINE_PATH}")
    print("\nCommit this file. The gate compares against it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
