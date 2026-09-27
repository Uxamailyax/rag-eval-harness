"""Build the Phase 6 golden set.

Fifty examples: 45 answerable ones taken straight from the benchmark, and 5
unanswerable ones constructed so their absence of an answer rests on evidence
rather than assertion.

**Why the benchmark supplies the answerable cases for free.** Each gold span carries
the exact contract text that answers its query. Nothing to write — the annotators
already did it, and Phase 1 verified we can read it correctly across 10,637 spans.

**Why the unanswerable cases are built by repointing, not by invention.** Proving an
answer *exists* is easy: point at it. Proving one *does not exist* means exhaustively
checking a 300 KB contract, and "I read it and it wasn't there" is a judgement rather
than evidence.

So instead: take a real query about contract A and repoint it at contract B, where
the benchmark records no annotated span of that question type.

**The catch this version fixes.** A first attempt accepted any target lacking the
question type, which produced targets annotated for as few as 2 question types. That
is nearly worthless as evidence: CUAD annotation is not exhaustive per document, so
absence there means "unobserved", not "absent".

Two changes make the evidence real:

  1. **Annotation density threshold.** A target must carry at least
     MIN_ANNOTATED_TYPES distinct annotated question types. If annotators labelled
     twenty different things in a document they clearly worked through it; a
     question type missing from that list means something. If they labelled two,
     absence proves nothing.

  2. **Natural prefixes.** The repointed query borrows its document description from
     a real query about the target, rather than pasting in a raw filename. Without
     this the five cases read as "Consider the document
     ArcaUsTreasuryFund_20200207_N-2_EX-99.K5_11971930..." — an obvious distribution
     shift that would make the retriever behave differently on them for reasons
     unrelated to answerability.

**Still not proof, and the write-up should say so.** Even at twenty annotated types
this is strong evidence of absence, not a guarantee. The honest claim is "the
benchmark's annotators labelled N distinct question types in this document and none
answers this question", not "this document contains no answer".

**Why unanswerable cases matter at all.** They are where hallucination becomes
visible. A model given context that does not contain the answer will usually answer
anyway. Without cases where the correct response is "I don't know", a faithfulness
metric has nothing to catch.

Run with:  uv run python scripts/build_golden_set.py
           uv run python scripts/build_golden_set.py --min-types 20
           uv run python scripts/build_golden_set.py --review
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

N_ANSWERABLE = 45
N_UNANSWERABLE = 5

# How thoroughly a document must be annotated before its lack of a question type
# counts as evidence. The first run produced targets with 2, 3 and 5 types — far too
# sparse to support the claim.
MIN_ANNOTATED_TYPES = 15

OUT_PATH = settings.data_dir.parent / "evals" / "golden_v1.jsonl"
PREFIX_SPLIT = ";"


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


def split_query(text: str) -> tuple[str, str]:
    """Return (prefix, question). Prefix names the document, question asks."""
    if PREFIX_SPLIT not in text:
        return "", text.strip()
    prefix, question = text.split(PREFIX_SPLIT, 1)
    return prefix.strip(), question.strip()


def question_signature(question: str) -> str:
    """A coarse fingerprint of what a question asks about.

    Used to establish that a target document has no answer of this *type*. Two
    queries about non-compete clauses should share a signature even when worded
    slightly differently.

    Deliberately crude. A false match costs nothing — the candidate is skipped. A
    false miss would silently weaken the unanswerable claim, so it errs toward
    matching.
    """
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    stop = {
        "does", "do", "did", "is", "are", "was", "were", "the", "this", "that",
        "a", "an", "of", "in", "on", "for", "to", "and", "or", "any", "there",
        "it", "its", "be", "been", "have", "has", "under", "contract", "agreement",
        "document", "what", "which", "who", "whom", "consider", "include",
        "included", "provide", "provided", "shall", "will", "can", "may",
    }
    words = [w for w in text.split() if w not in stop and len(w) > 2]
    return " ".join(sorted(set(words)))


def build_unanswerable(
    source: LegalBenchRAG,
    all_queries: list[Query],
    exclude_ids: set[str],
    n: int,
    seed: int,
    min_types: int,
) -> tuple[list[dict], dict]:
    """Repoint real queries at densely-annotated documents that lack that question type."""
    rng = random.Random(seed + 1)

    # Annotated question types per document, and a natural prefix for each document
    # borrowed from a real query about it.
    doc_signatures: dict[str, set[str]] = defaultdict(set)
    doc_prefixes: dict[str, str] = {}
    doc_subset: dict[str, str] = {}

    for q in all_queries:
        prefix, question = split_query(q.text)
        sig = question_signature(question)
        for span in q.spans:
            doc_signatures[span.doc_id].add(sig)
            doc_subset[span.doc_id] = q.subset
            if prefix and span.doc_id not in doc_prefixes:
                doc_prefixes[span.doc_id] = prefix

    # Documents dense enough for absence to mean something.
    dense_docs = {
        d for d, sigs in doc_signatures.items()
        if len(sigs) >= min_types and source.has_document(d) and d in doc_prefixes
    }

    by_subset: dict[str, list[str]] = defaultdict(list)
    for d in dense_docs:
        by_subset[doc_subset[d]].append(d)

    stats = {
        "documents_total": len(doc_signatures),
        "documents_dense_enough": len(dense_docs),
        "min_annotated_types": min_types,
        "dense_by_subset": {s: len(v) for s, v in sorted(by_subset.items())},
        "max_types_seen": max((len(v) for v in doc_signatures.values()), default=0),
    }

    pool = [q for q in all_queries if q.query_id not in exclude_ids]
    rng.shuffle(pool)

    built: list[dict] = []
    used_targets: set[str] = set()

    for query in pool:
        if len(built) >= n:
            break

        _, question = split_query(query.text)
        sig = question_signature(question)
        gold_docs = {s.doc_id for s in query.spans}

        candidates = [
            d
            for d in by_subset.get(query.subset, [])
            if d not in gold_docs and d not in used_targets and sig not in doc_signatures[d]
        ]
        if not candidates:
            continue

        # Prefer the most densely annotated candidate: the stronger the evidence of
        # thorough annotation, the stronger the claim of absence.
        target = max(sorted(candidates), key=lambda d: len(doc_signatures[d]))
        n_types = len(doc_signatures[target])

        built.append(
            {
                "id": f"unanswerable:{len(built):02d}",
                "source_query_id": query.query_id,
                "subset": query.subset,
                # A natural prefix borrowed from a real query about the target,
                # so these five read like every other query in the set.
                "question": f"{doc_prefixes[target]}{PREFIX_SPLIT} {question}",
                "answerable": False,
                "expected_answer": None,
                "target_doc": target,
                "target_annotated_types": n_types,
                "original_gold_docs": sorted(gold_docs),
                "question_signature": sig,
                "evidence": (
                    f"The benchmark's annotators labelled {n_types} distinct question "
                    f"types as having answers in this document, and this question's "
                    f"type is not among them. Strong evidence of absence — not proof, "
                    f"since annotation is not guaranteed exhaustive."
                ),
            }
        )
        used_targets.add(target)

    return built, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-types", type=int, default=MIN_ANNOTATED_TYPES)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()

    source = LegalBenchRAG(settings.data_dir)
    usable = source.usable_queries()
    print(f"usable queries in benchmark : {len(usable)}")

    # --- answerable ---------------------------------------------------------
    answerable_queries = stratified_sample(usable, N_ANSWERABLE, settings.random_seed)
    answerable: list[dict] = []

    for query in answerable_queries:
        spans = []
        for span in query.spans:
            spans.append(
                {
                    "doc_id": span.doc_id,
                    "start": span.start,
                    "end": span.end,
                    "text": source.document_text(span.doc_id)[span.start : span.end],
                }
            )
        answerable.append(
            {
                "id": f"answerable:{len(answerable):02d}",
                "source_query_id": query.query_id,
                "subset": query.subset,
                "question": query.text,
                "answerable": True,
                # The reference answer is the annotators' own span text. Nothing
                # was written by hand.
                "expected_answer": " ".join(s["text"].strip() for s in spans),
                "gold_spans": spans,
                "gold_chars": sum(s["end"] - s["start"] for s in spans),
            }
        )

    counts = {
        s: sum(1 for a in answerable if a["subset"] == s)
        for s in sorted({a["subset"] for a in answerable})
    }
    print(f"answerable cases            : {len(answerable)}  {counts}")

    # --- unanswerable -------------------------------------------------------
    unanswerable, stats = build_unanswerable(
        source,
        usable,
        exclude_ids={q.query_id for q in answerable_queries},
        n=N_UNANSWERABLE,
        seed=settings.random_seed,
        min_types=args.min_types,
    )

    print(f"\n--- annotation density ---")
    print(f"  documents referenced by the benchmark : {stats['documents_total']}")
    print(f"  most annotated question types in one  : {stats['max_types_seen']}")
    print(f"  threshold for a usable target         : >= {args.min_types} types")
    print(f"  documents dense enough                : {stats['documents_dense_enough']}")
    print(f"  by subset                             : {stats['dense_by_subset']}")

    print(f"\nunanswerable cases          : {len(unanswerable)}")
    if len(unanswerable) < N_UNANSWERABLE:
        print(f"  WARNING: wanted {N_UNANSWERABLE}, built {len(unanswerable)}.")
        print(f"  Lower --min-types, at the cost of weaker evidence.")

    golden = answerable + unanswerable
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for row in golden:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwritten: {OUT_PATH}  ({len(golden)} examples)")

    # --- the five to read by hand -------------------------------------------
    print(f"\n{'=' * 90}")
    print("READ THESE FIVE — the unanswerable cases")
    print("=" * 90)
    print("\nThe construction is automatic; the claim is only as good as the cases.")
    print("For each, confirm: the question is sensible, the target is the right kind")
    print("of document, and the answer genuinely is not there.\n")

    for case in unanswerable:
        print(f"  [{case['id']}]  repointed from {case['source_query_id']} "
              f"({case['subset']})")
        print(f"    question    : {case['question'][:170]}")
        print(f"    target doc  : {case['target_doc'].split('/')[-1][:70]}")
        print(f"    annotated   : {case['target_annotated_types']} question types "
              f"labelled in this document")
        print(f"    originally  : {case['original_gold_docs'][0].split('/')[-1][:70]}")
        print()

    if args.review:
        print(f"{'=' * 90}")
        print("ALL ANSWERABLE CASES")
        print("=" * 90)
        for case in answerable:
            print(f"\n  [{case['id']}] {case['subset']}  ({case['gold_chars']} gold chars)")
            print(f"    Q: {case['question'][:170]}")
            answer = " ".join(case["expected_answer"].split())
            print(f"    A: {answer[:220]}{'...' if len(answer) > 220 else ''}")

    # --- gate ----------------------------------------------------------------
    weakest = min((c["target_annotated_types"] for c in unanswerable), default=0)

    print(f"\n{'=' * 90}")
    print("--- gate ---")
    checks = {
        "50 examples total": len(golden) == N_ANSWERABLE + N_UNANSWERABLE,
        f"{N_UNANSWERABLE} unanswerable cases": len(unanswerable) == N_UNANSWERABLE,
        "every answerable case has a reference answer": all(
            a["expected_answer"] for a in answerable
        ),
        "every unanswerable case targets a different document": (
            len({u["target_doc"] for u in unanswerable}) == len(unanswerable)
        ),
        "all four subsets represented in the answerable half": (
            len({a["subset"] for a in answerable}) == 4
        ),
        f"weakest evidence >= {args.min_types} annotated types": weakest >= args.min_types,
        "no unanswerable question names a raw filename": all(
            not u["question"].startswith("Consider the document ")
            for u in unanswerable
        ),
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\n  weakest case rests on {weakest} annotated question types")

    print("\n--- manual gate (you decide) ---")
    print("  [ ] You read all five and agree they are unanswerable")
    print("  [ ] You spot-read ~10 answerable cases and the gold text answers the question")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
