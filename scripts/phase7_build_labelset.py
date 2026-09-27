"""Phase 7, step 1: build the human-labelling set.

Thirty cases for a human to label faithful or unfaithful. Those labels become the
ground truth against which the LLM judge is measured.

**These thirty are not the point. The judge is.**

The thirty answers are already known — they were read and labelled by hand. What is
unknown is whether the judge can be trusted on the other twenty, and on every answer
the system produces afterwards that nobody will ever label. The thirty are a test set
for the grader:

    kappa = 0.85   the judge agrees where the answer is known, so trust it where it is not
    kappa = 0.20   the judge is close to guessing, and every faithfulness score is noise

Same pattern as Phase 2, where the metrics were proved on toy data with known answers
before being trusted on real data. This is that, applied to a judge rather than a
formula.

**Why the set must contain unfaithful cases.**

Phase 6 found the generator did not hallucinate once in fifty questions. Good for the
system, fatal for the measurement: if all thirty are faithful, the human says "yes"
thirty times, the judge says "yes" thirty times, and agreement is 100%.

But expected agreement by chance is also ~100% — two raters who always say "yes" agree
every time even at random. So:

    kappa = (1.00 - 1.00) / (1 - 1.00) = 0 / 0

Undefined, or near-zero in practice. **100% agreement, kappa around zero.** The kappa
paradox. Agreeing that everything is faithful proves nothing, because a judge that
outputs "faithful" unconditionally would score identically.

Kappa only becomes informative when both kinds of case are present and the judge has to
actually discriminate. So the set is stratified:

    ~10 grounded answers    real answers where retrieval found gold
    ~10 refusals            "I don't know" — an edge case worth deciding explicitly
    ~10 corrupted           real answers paired with mismatched context, so genuinely
                            unfaithful cases exist to detect

**How the corrupted cases are made.** Take a real answer and pair it with a different
question's retrieved context. The answer is fluent, specific and quotes real contract
language — it simply describes a document the context does not contain. That is exactly
the hallucination profile Phase 6b identified as the dangerous one: not invention from
nothing, but confident assertion unsupported by the text provided.

This is the plan's "induced hallucination", and it is necessary rather than optional
because the system refused to produce any naturally.

**The refusal question.** Is "I don't know" faithful? It asserts nothing unsupported,
so the answer here is yes — but it has to be decided before labelling starts and applied
consistently, because a judge that disagrees on this one class could tank kappa for a
reason that has nothing to do with hallucination detection.

Run with:  uv run python scripts/phase7_build_labelset.py
           uv run python scripts/phase7_build_labelset.py --print
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys

from rag_eval_harness.config import settings

EVALS = settings.data_dir.parent / "evals"
ANSWERS_PATH = EVALS / "answers_v2.jsonl"
LABELSET_PATH = EVALS / "labelset_v1.jsonl"
BLANK_LABELS_PATH = EVALS / "human_labels_v1_BLANK.jsonl"

N_GROUNDED = 10
N_REFUSAL = 10
N_CORRUPTED = 10

REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bexcerpts? (do|does) not (contain|include|provide|mention|address)\b",
    r"\bexcerpts? contain no\b",
    r"\b(does|do) not contain any provision\b",
    r"\bthere is no (provision|clause|mention|reference|language)\b",
    r"\bcannot (be )?(answer|determine|establish|tell)\b",
    r"\bunable to (answer|determine|find)\b",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def is_refusal(answer: str) -> bool:
    return any(p.search(answer) for p in COMPILED)


def load_answers() -> list[dict]:
    if not ANSWERS_PATH.exists():
        raise FileNotFoundError(
            f"missing {ANSWERS_PATH}\n"
            "Run: uv run python scripts/phase6_generate.py --golden golden_v2.jsonl"
        )
    with open(ANSWERS_PATH, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="print all 30 cases in full for reading")
    args = parser.parse_args()

    answers = load_answers()
    rng = random.Random(settings.random_seed + 7)

    for a in answers:
        a["_refusal"] = is_refusal(a["answer"])

    grounded_pool = [
        a for a in answers
        if a["answerable"] and not a["_refusal"] and a["retrieval_recall_10"] > 0
    ]
    refusal_pool = [a for a in answers if a["_refusal"]]
    # Answers that assert something, used as raw material for corruption.
    assertive_pool = [a for a in answers if not a["_refusal"]]

    print(f"answers available   : {len(answers)}")
    print(f"  grounded (recall>0, asserted) : {len(grounded_pool)}")
    print(f"  refusals                      : {len(refusal_pool)}")
    print(f"  assertive (corruption source) : {len(assertive_pool)}\n")

    cases: list[dict] = []

    # --- grounded: expected faithful ----------------------------------------
    for a in rng.sample(grounded_pool, min(N_GROUNDED, len(grounded_pool))):
        cases.append(
            {
                "case_id": f"case:{len(cases):02d}",
                "stratum": "grounded",
                "source_answer_id": a["id"],
                "question": a["question"],
                "answer": a["answer"],
                # The context shown to the labeller and the judge is the context
                # the answer was actually generated from.
                "context_source_id": a["id"],
                "retrieved": a["retrieved"],
                "retrieval_recall_10": a["retrieval_recall_10"],
                "corrupted": False,
                "note": "real answer, generated from this context",
            }
        )

    # --- refusals: the edge case --------------------------------------------
    for a in rng.sample(refusal_pool, min(N_REFUSAL, len(refusal_pool))):
        cases.append(
            {
                "case_id": f"case:{len(cases):02d}",
                "stratum": "refusal",
                "source_answer_id": a["id"],
                "question": a["question"],
                "answer": a["answer"],
                "context_source_id": a["id"],
                "retrieved": a["retrieved"],
                "retrieval_recall_10": a["retrieval_recall_10"],
                "corrupted": False,
                "note": "a refusal — decide once whether refusing counts as faithful",
            }
        )

    # --- corrupted: induced hallucination -----------------------------------
    # An answer from one question paired with another question's context. The
    # answer stays fluent and specific and quotes real contract language; it simply
    # describes a document that is not in front of the judge.
    used_pairs: set[tuple[str, str]] = set()
    corrupted = 0
    attempts = 0

    while corrupted < N_CORRUPTED and attempts < 500:
        attempts += 1
        answer_src = rng.choice(assertive_pool)
        context_src = rng.choice(answers)

        if answer_src["id"] == context_src["id"]:
            continue
        # Reject pairs that share a source document: the answer might then be
        # genuinely supported, which would put a wrong label in the ground truth.
        answer_docs = {h["doc_id"] for h in answer_src["retrieved"]}
        context_docs = {h["doc_id"] for h in context_src["retrieved"]}
        if answer_docs & context_docs:
            continue
        pair = (answer_src["id"], context_src["id"])
        if pair in used_pairs:
            continue
        used_pairs.add(pair)

        cases.append(
            {
                "case_id": f"case:{len(cases):02d}",
                "stratum": "corrupted",
                "source_answer_id": answer_src["id"],
                # The question shown is the one the *context* belongs to, so the
                # case reads like a normal question-context-answer triple.
                "question": context_src["question"],
                "answer": answer_src["answer"],
                "context_source_id": context_src["id"],
                "retrieved": context_src["retrieved"],
                "retrieval_recall_10": context_src["retrieval_recall_10"],
                "corrupted": True,
                "note": (
                    f"answer taken from {answer_src['id']}, context from "
                    f"{context_src['id']} — no shared source documents"
                ),
            }
        )
        corrupted += 1

    if corrupted < N_CORRUPTED:
        print(f"WARNING: built {corrupted} corrupted cases of {N_CORRUPTED}")

    # Shuffle so the labeller cannot infer a label from position.
    rng.shuffle(cases)
    for i, case in enumerate(cases):
        case["case_id"] = f"case:{i:02d}"

    with open(LABELSET_PATH, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    # A blank file to fill in, carrying no hint of the stratum.
    with open(BLANK_LABELS_PATH, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "faithful": None,
                        "note": "",
                    }
                )
                + "\n"
            )

    strata: dict[str, int] = {}
    for c in cases:
        strata[c["stratum"]] = strata.get(c["stratum"], 0) + 1

    print(f"labelling set : {len(cases)} cases")
    for name, count in sorted(strata.items()):
        print(f"  {name:<12}{count:>4}")

    print(f"\n  written: {LABELSET_PATH}")
    print(f"  blanks : {BLANK_LABELS_PATH}")

    # --- how to label --------------------------------------------------------
    print(f"\n{'=' * 84}")
    print("HOW TO LABEL")
    print("=" * 84)
    print("""
  The question to answer for each case is narrow, and it is not "is this answer
  correct?":

      Is every claim in this answer supported by the excerpts shown?

  faithful = 1    every factual claim traces to the excerpts
  faithful = 0    the answer asserts something the excerpts do not support

  An answer can be factually true about the real world and still unfaithful, if the
  support is not in the context provided. Faithfulness is about grounding, not truth.

  Two rules to fix before starting, and apply to every case:

    1. A refusal ("I don't know") asserts nothing unsupported, so it is faithful.
       Decide this once. Changing position halfway makes the labels inconsistent
       and the kappa meaningless.

    2. Judge only what is in front of you. Do not open the source document to check
       whether a claim is true — the judge cannot do that either, and the two raters
       must be answering the same question.

  Fill in `faithful` in the blank file with 1 or 0. The `note` field is for anything
  that made a case difficult; those notes are worth more than the labels when the
  kappa comes out low.

  The strata are deliberately hidden from the blank file. Roughly a third of the set
  was constructed to be unfaithful, but which third is not marked.
""")

    if args.do_print:
        print(f"{'=' * 84}")
        print("ALL 30 CASES")
        print("=" * 84)
        for case in cases:
            print(f"\n{'-' * 84}")
            print(f"[{case['case_id']}]")
            print(f"\nQUESTION:\n  {case['question']}")
            print(f"\nANSWER:\n  {' '.join(case['answer'].split())}")
            print(f"\nCONTEXT — {len(case['retrieved'])} excerpts retrieved:")
            for h in case["retrieved"][:10]:
                print(f"  {h['rank']}. {h['doc_id'].split('/')[-1][:70]}")

    print(f"\n  To read the cases with their full excerpt text:")
    print(f"    uv run python scripts/phase7_show_case.py case:00")

    return 0


if __name__ == "__main__":
    sys.exit(main())
