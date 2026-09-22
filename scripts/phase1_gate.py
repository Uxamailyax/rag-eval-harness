"""Phase 1 gate: verify the ground truth can be read correctly.

For every annotation, slice the corpus file at the gold span and compare against
the `answer` string the dataset stores. Matching means our file reading agrees
with the annotators' and the offsets are trustworthy. Not matching means every
downstream metric is measuring noise.

The gate separates two things that look alike but are not:

  - a span that resolves to a file and returns the wrong text -> a reading bug
  - a span whose file is not on disk at all -> a coverage gap

The first invalidates the benchmark. The second just shrinks it, and is reported
as a coverage number rather than a failure, provided it stays small.

Run with:  uv run python scripts/phase1_gate.py
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from collections import Counter

from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG

PASS_THRESHOLD = 0.98  # of spans whose document exists
MIN_COVERAGE = 0.95  # of all spans, documents must be reachable
SAMPLE_MISMATCHES = 5
MANUAL_REVIEW_SAMPLE = 5


def classify(sliced: str, expected: str) -> str:
    """Describe how a mismatch differs. The pattern points at the cause."""
    if sliced == expected:
        return "exact"
    if sliced.strip() == expected.strip():
        return "whitespace_only"
    if unicodedata.normalize("NFC", sliced) == unicodedata.normalize("NFC", expected):
        return "unicode_normalisation"
    if len(sliced) == len(expected):
        # Same window size, different content: the window is in the wrong place.
        return "offset_shift"
    if expected and expected in sliced:
        return "expected_is_substring"
    if sliced and sliced in expected:
        return "sliced_is_substring"
    return "different_text"


def show(label: str, text: str, limit: int = 120) -> None:
    body = text[:limit].replace("\n", "\\n").replace("\r", "\\r")
    suffix = "..." if len(text) > limit else ""
    print(f"    {label:9} ({len(text):6} chars): {body}{suffix}")


def main() -> int:
    source = LegalBenchRAG(settings.data_dir)

    print(f"corpus     : {source.corpus_dir}")
    print(f"benchmarks : {source.benchmarks_dir}\n")

    queries = source.queries()
    docs = source.document_ids()
    total_spans = sum(len(q.spans) for q in queries)

    print(f"documents on disk : {len(docs)}")
    print(f"queries in JSON   : {len(queries)}")
    print(f"gold spans        : {total_spans}")
    print(f"per subset        : {source.subset_counts()}\n")

    # --- coverage: which referenced documents actually exist ---------------
    print("--- document coverage ---")
    referenced = {s.doc_id for q in queries for s in q.spans}
    unreachable = sorted(d for d in referenced if not source.has_document(d))
    illegal = [d for d in unreachable if source.is_unreachable_on_windows(d)]
    other_missing = [d for d in unreachable if d not in set(illegal)]

    print(f"  documents referenced by queries : {len(referenced)}")
    print(f"  not found on disk               : {len(unreachable)}")
    print(f"    - illegal Windows filename    : {len(illegal)}")
    print(f"    - missing for another reason  : {len(other_missing)}")
    for doc_id in other_missing[:5]:
        print(f"        {doc_id}")

    usable = source.usable_queries()
    dropped = len(queries) - len(usable)
    usable_spans = sum(len(q.spans) for q in usable)
    coverage = usable_spans / total_spans if total_spans else 0.0

    print(f"\n  usable queries : {len(usable)}  (dropped {dropped})")
    print(f"  usable spans   : {usable_spans} / {total_spans}  ({coverage:.2%})")
    print(f"  per subset     : {source.subset_counts(usable)}")

    # --- verify every span whose document exists ---------------------------
    print("\n--- verifying spans ---")
    outcomes: Counter[str] = Counter()
    missing_answer = 0
    out_of_range = 0
    mismatches: list[dict[str, object]] = []

    for query in usable:
        for span in query.spans:
            if span.answer is None:
                missing_answer += 1
                continue

            doc = source.document_text(span.doc_id)
            if span.end > len(doc):
                out_of_range += 1
                outcomes["span_out_of_range"] += 1
                continue

            sliced = doc[span.start : span.end]
            kind = classify(sliced, span.answer)
            outcomes[kind] += 1
            if kind != "exact" and len(mismatches) < 200:
                mismatches.append(
                    {
                        "query_id": query.query_id,
                        "doc_id": span.doc_id,
                        "span": [span.start, span.end],
                        "kind": kind,
                        "sliced": sliced,
                        "expected": span.answer,
                    }
                )

    checked = sum(outcomes.values())
    exact = outcomes["exact"]
    pass_rate = exact / checked if checked else 0.0

    print(f"  checked : {checked}")
    print(f"  exact   : {exact}  ({pass_rate:.2%})\n")
    print("  outcome breakdown:")
    for kind, count in outcomes.most_common():
        print(f"    {kind:24} {count:6}  ({count / checked:.2%})")
    if missing_answer:
        print(f"\n  (skipped {missing_answer} spans with no stored answer)")

    if mismatches:
        print(f"\n--- sample mismatches ({min(SAMPLE_MISMATCHES, len(mismatches))}) ---")
        for item in mismatches[:SAMPLE_MISMATCHES]:
            print(f"\n  [{item['kind']}] {item['query_id']}  {item['doc_id']}")
            print(f"    span {item['span']}")
            show("expected", str(item["expected"]))
            show("sliced", str(item["sliced"]))

    # --- a few passing examples to read by eye ------------------------------
    print(f"\n--- {MANUAL_REVIEW_SAMPLE} passing examples for manual review ---")
    rng = random.Random(settings.random_seed)
    verified = [
        (q, s)
        for q in usable
        for s in q.spans
        if s.answer and source.slice_span(s) == s.answer
    ]
    if verified:
        for query, span in rng.sample(verified, min(MANUAL_REVIEW_SAMPLE, len(verified))):
            print(f"\n  {query.query_id} [{query.subset}]")
            print(f"    Q: {query.text[:150]}")
            show("A", str(source.slice_span(span)))

    # --- gate --------------------------------------------------------------
    checks = {
        f"span verification >= {PASS_THRESHOLD:.0%} (of reachable spans)": (
            pass_rate >= PASS_THRESHOLD
        ),
        f"document coverage >= {MIN_COVERAGE:.0%} (of all spans)": coverage >= MIN_COVERAGE,
        "no spans out of document range": out_of_range == 0,
        "all four subsets survive filtering": len(source.subset_counts(usable)) == 4,
    }

    print("\n--- gate ---")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    passed = all(checks.values())

    manifest_path = write_manifest(
        settings,
        {
            "phase": 1,
            "gate": "span_verification",
            "passed": passed,
            "documents_on_disk": len(docs),
            "queries_total": len(queries),
            "queries_usable": len(usable),
            "spans_total": total_spans,
            "spans_usable": usable_spans,
            "coverage": round(coverage, 4),
            "spans_checked": checked,
            "exact_matches": exact,
            "pass_rate": round(pass_rate, 4),
            "outcomes": dict(outcomes),
            "unreachable_documents": len(unreachable),
            "unreachable_illegal_filename": len(illegal),
            "subset_counts_usable": source.subset_counts(usable),
        },
    )

    if unreachable:
        (manifest_path.parent / "unreachable_documents.json").write_text(
            json.dumps(unreachable, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if mismatches:
        report = manifest_path.parent / "span_mismatches.json"
        report.write_text(
            json.dumps(mismatches, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nmismatch detail: {report}")

    print("\nPHASE 1 GATE: PASS" if passed else "\nPHASE 1 GATE: FAIL")
    if not passed:
        print(
            "\nDo not proceed. Read the outcome breakdown:\n"
            "  offset_shift          -> the read and the annotation disagree on\n"
            "                           character positions; suspect newline or\n"
            "                           encoding handling\n"
            "  whitespace_only       -> harmless, relax the comparison\n"
            "  unicode_normalisation -> harmless, normalise both sides\n"
            "  different_text        -> offsets genuinely do not align; consider\n"
            "                           the ACORD fallback corpus"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
