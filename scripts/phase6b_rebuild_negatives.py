"""Phase 6b: rebuild the unanswerable cases as provable negatives.

Phase 6a's unanswerable cases rested on an assumption that does not hold. They were
built by repointing a contract question at a different contract where the benchmark
recorded no annotated span of that question type. But CUAD uses a fixed 41-category
schema, so a document annotated for 19 categories was only ever *checked* for those
19. The other 22 were never examined.

    Annotators claimed these answers are here. They never claimed no other answers
    are here — and that claim is not feasible to make about a 300 KB contract.

Reading the five, several looked correct: one named a specific exclusivity section
and quoted its carve-outs. Absence of a label is not absence of a clause, so "5
hallucinations" was never established.

**The fix: cross-domain repointing.** Instead of pointing a contract question at
another contract, point it at a document of a different *type* — one that
structurally cannot contain the answer.

    "What is the governing law of this contract?"  ->  a mobile app privacy policy

A privacy policy has no parties, no governing-law clause, no termination terms, no IP
assignment. Not because annotators missed them, but because that document type does
not contain them. The absence is **structural** rather than unobserved, and it
requires no reading to verify.

| | Phase 6a | Phase 6b |
|---|---|---|
| repointing | contract -> different contract | contract -> different document type |
| absence is | unobserved | structural |
| rests on | annotation coverage | what the document type is |
| test difficulty | harder | easier |
| scoreable | no | yes |

**The trade is deliberate.** Cross-domain mismatches are more obvious, so the test is
easier — a model may refuse simply because the context is visibly unrelated. But a
hard test you cannot score is worth less than an easy test you can. Phase 6a already
established what happens under the *harder* condition; what it could not establish is
whether those answers were wrong.

**The pairings, and why each is impossible**

    contract question   -> privacy policy   no parties, no clauses, not a contract
    merger question     -> NDA              only merger agreements define MAE,
                                            termination fees, appraisal rights
    privacy question    -> contract         contracts do not describe data collection

Run with:  uv run python scripts/phase6b_rebuild_negatives.py
           uv run python scripts/phase6b_rebuild_negatives.py --review
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict

from rag_eval_harness.config import settings
from rag_eval_harness.ground_truth.base import Query
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG

GOLDEN_IN = settings.data_dir.parent / "evals" / "golden_v1.jsonl"
GOLDEN_OUT = settings.data_dir.parent / "evals" / "golden_v2.jsonl"

N_NEGATIVES = 5
PREFIX_SPLIT = ";"

# Which subsets can host a question from which other subset, and the structural
# reason the answer cannot be there. Ordered so the strongest mismatches come first.
#
# privacy_qa documents are the strongest targets in the corpus: they are not
# contracts at all, so no contractual question of any kind can be answered by one.
PAIRINGS = [
    (
        "cuad",
        "privacy_qa",
        "A commercial contract question pointed at a mobile app privacy policy. "
        "A privacy policy is a website's data-handling notice — it has no parties, "
        "no clauses, no governing law, no termination terms. It is not a contract "
        "between parties at all, so no question about contractual terms can be "
        "answered by one.",
    ),
    (
        "maud",
        "privacy_qa",
        "A merger-agreement question pointed at a mobile app privacy policy. "
        "Material Adverse Effect definitions, termination fees and appraisal rights "
        "exist only in merger agreements. A privacy policy contains none of these "
        "concepts in any form.",
    ),
    (
        "contractnli",
        "privacy_qa",
        "An NDA question pointed at a mobile app privacy policy. Confidentiality "
        "obligations between a Disclosing and a Receiving Party require two parties "
        "to an agreement. A privacy policy is a unilateral notice to users and has "
        "no such parties.",
    ),
    (
        "maud",
        "contractnli",
        "A merger-agreement question pointed at a non-disclosure agreement. MAE "
        "definitions, deal-protection terms and closing conditions belong to merger "
        "agreements; an NDA governs confidentiality only and contains no acquisition "
        "mechanics.",
    ),
    (
        "privacy_qa",
        "cuad",
        "A privacy-policy question pointed at a commercial contract. Questions about "
        "what user data an app collects or sells cannot be answered by a contract "
        "between two companies, which describes obligations rather than data "
        "practices.",
    ),
]


def split_query(text: str) -> tuple[str, str]:
    if PREFIX_SPLIT not in text:
        return "", text.strip()
    prefix, question = text.split(PREFIX_SPLIT, 1)
    return prefix.strip(), question.strip()


def describe_document(doc_id: str, subset: str, prefixes: dict[str, str]) -> str:
    """A natural description of the target, for the rewritten query prefix.

    Borrowed from a real query about that document where one exists, so the five
    read like every other query in the set rather than naming a raw filename.
    """
    if doc_id in prefixes:
        return prefixes[doc_id]
    name = doc_id.split("/")[-1].rsplit(".", 1)[0]
    if subset == "privacy_qa":
        return f'Consider "{name}"\'s privacy policy'
    return f"Consider the {name} agreement"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()

    if not GOLDEN_IN.exists():
        print(f"missing {GOLDEN_IN}\nRun scripts/build_golden_set.py first.")
        return 1

    with open(GOLDEN_IN, "r", encoding="utf-8") as fh:
        existing = [json.loads(line) for line in fh if line.strip()]

    answerable = [g for g in existing if g["answerable"]]
    old_negatives = [g for g in existing if not g["answerable"]]

    print(f"existing golden set : {len(existing)}")
    print(f"  answerable        : {len(answerable)} (kept unchanged)")
    print(f"  unanswerable      : {len(old_negatives)} (being replaced)\n")

    source = LegalBenchRAG(settings.data_dir)
    usable = source.usable_queries()

    # Documents by subset, and a natural prefix for each where one exists.
    docs_by_subset: dict[str, set[str]] = defaultdict(set)
    prefixes: dict[str, str] = {}
    for q in usable:
        prefix, _ = split_query(q.text)
        for span in q.spans:
            docs_by_subset[q.subset].add(span.doc_id)
            if prefix and span.doc_id not in prefixes:
                prefixes[span.doc_id] = prefix

    print("--- corpus by subset ---")
    for subset in sorted(docs_by_subset):
        n_q = sum(1 for q in usable if q.subset == subset)
        print(f"  {subset:<14}{len(docs_by_subset[subset]):>5} documents"
              f"{n_q:>7} queries")

    queries_by_subset: dict[str, list[Query]] = defaultdict(list)
    for q in usable:
        queries_by_subset[q.subset].append(q)

    rng = random.Random(settings.random_seed + 2)
    negatives: list[dict] = []
    used_targets: set[str] = set()

    print(f"\n--- building {N_NEGATIVES} provable negatives ---\n")

    for source_subset, target_subset, rationale in PAIRINGS:
        if len(negatives) >= N_NEGATIVES:
            break

        pool = sorted(queries_by_subset.get(source_subset, []), key=lambda q: q.query_id)
        targets = sorted(docs_by_subset.get(target_subset, set()) - used_targets)
        if not pool or not targets:
            print(f"  skipped {source_subset} -> {target_subset}: nothing available")
            continue

        query = rng.choice(pool)
        target = rng.choice(targets)
        _, question = split_query(query.text)

        negatives.append(
            {
                "id": f"unanswerable:{len(negatives):02d}",
                "source_query_id": query.query_id,
                "source_subset": source_subset,
                "subset": source_subset,
                "target_subset": target_subset,
                "question": f"{describe_document(target, target_subset, prefixes)}"
                            f"{PREFIX_SPLIT} {question}",
                "original_question": query.text,
                "answerable": False,
                "expected_answer": None,
                "target_doc": target,
                "original_gold_docs": sorted({s.doc_id for s in query.spans}),
                "construction": "cross_domain_repointing",
                "evidence": rationale,
                "evidence_type": "structural",
            }
        )
        used_targets.add(target)

        print(f"  [{negatives[-1]['id']}] {source_subset} question -> "
              f"{target_subset} document")
        print(f"     Q: {question[:110]}")
        print(f"     target: {target.split('/')[-1][:60]}")
        print()

    if len(negatives) < N_NEGATIVES:
        print(f"  WARNING: built {len(negatives)} of {N_NEGATIVES}")

    golden = answerable + negatives
    with open(GOLDEN_OUT, "w", encoding="utf-8") as fh:
        for row in golden:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"written: {GOLDEN_OUT}  ({len(golden)} examples)")

    # --- read these ----------------------------------------------------------
    print(f"\n{'=' * 88}")
    print("READ THESE FIVE")
    print("=" * 88)
    print("\nEach should be obviously impossible once you see the pairing. If any")
    print("requires thought, the mismatch is not structural enough.\n")

    for case in negatives:
        print(f"  [{case['id']}]  {case['source_subset']} -> {case['target_subset']}")
        print(f"    question : {case['question'][:165]}")
        print(f"    target   : {case['target_doc']}")
        print(f"    why       : {case['evidence'][:200]}")
        print()

    # --- what changed from 6a ------------------------------------------------
    print(f"{'=' * 88}")
    print("WHAT CHANGED FROM PHASE 6a")
    print("=" * 88)
    print(f"\n  {'':<18}{'6a':<34}{'6b':<34}")
    print("  " + "-" * 84)
    rows = [
        ("repointing", "contract -> other contract", "contract -> other doc type"),
        ("absence is", "unobserved", "structural"),
        ("rests on", "annotation coverage", "what the document type is"),
        ("verifiable by", "trusting CUAD's 41 categories", "knowing what a privacy policy is"),
        ("test difficulty", "harder", "easier"),
        ("scoreable", "no", "yes"),
    ]
    for label, a, b in rows:
        print(f"  {label:<18}{a:<34}{b:<34}")

    print("\n  The trade is deliberate. Cross-domain mismatches are more obvious, so a")
    print("  model may refuse simply because the context is visibly unrelated. Phase 6a")
    print("  already measured the harder condition; what it could not establish is")
    print("  whether those answers were actually wrong.")

    if args.review:
        print(f"\n{'=' * 88}")
        print("THE 6a NEGATIVES BEING REPLACED")
        print("=" * 88)
        for case in old_negatives:
            print(f"\n  [{case['id']}] target had "
                  f"{case.get('target_annotated_types', '?')} annotated types")
            print(f"    {case['question'][:150]}")

    # --- gate ----------------------------------------------------------------
    print(f"\n{'=' * 88}")
    print("--- gate ---")
    checks = {
        f"{N_NEGATIVES} negatives built": len(negatives) == N_NEGATIVES,
        "every negative crosses subsets": all(
            c["source_subset"] != c["target_subset"] for c in negatives
        ),
        "every negative targets a different document": (
            len({c["target_doc"] for c in negatives}) == len(negatives)
        ),
        "45 answerable cases carried over unchanged": len(answerable) == 45,
        "50 examples total": len(golden) == 50,
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] Each pairing is obviously impossible, not merely unlikely")
    print("  [ ] None of the five needs a document read to be confident about")

    print(f"\nnext: uv run python scripts/phase6_generate.py --golden golden_v2.jsonl")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
