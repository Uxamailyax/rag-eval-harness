"""Strip corpus text from the evaluation files before publication.

Phase 1 established that the corpus cannot be redistributed: the LegalBench-RAG
repository is MIT, but the underlying documents come from ContractNLI, CUAD, MAUD and
PrivacyQA, each with its own usage policy, and redistribution rights could not be
confirmed. The decision recorded then was **publish metrics, IDs and hashes, never the
corpus.**

Several evaluation files violate that without meaning to. `golden_v2.jsonl` carries
27,080 characters of verbatim contract text in its `expected_answer` and span `text`
fields; `labelset_v1.jsonl` carries generated answers that quote contracts directly.

**Nothing is lost by removing them.** Every gold span is stored as
`(doc_id, start, end)`, and that is how the harness itself reads them — Phase 1
verified 10,637 spans resolve correctly from offsets alone. Anyone who downloads the
corpus can reconstruct every character. The offsets are the reproducible artifact; the
text is a convenience copy.

**What survives sanitisation**

    doc_id, start, end        the span, exactly as the benchmark defines it
    question                  the query, which is the benchmark's own text
    retrieval metadata        ranks, scores, which chunks were sent
    scores and labels         every number the results depend on

**What is removed**

    expected_answer           verbatim contract text
    gold_spans[].text         verbatim contract text
    answer                    generated text quoting contracts verbatim

A `reconstruct` mode restores the text locally from the corpus, so a working copy
loses nothing either.

Run with:  uv run python scripts/sanitise_evals.py --check
           uv run python scripts/sanitise_evals.py --sanitise
           uv run python scripts/sanitise_evals.py --reconstruct
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rag_eval_harness.config import PROJECT_ROOT, settings
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG

EVALS = PROJECT_ROOT / "evals"

# Fields that carry verbatim corpus text, by file. Everything else is metadata,
# numbers, or the benchmark's own query text.
TEXT_FIELDS = {
    "golden_v1.jsonl": {"top": ["expected_answer"], "spans": "gold_spans"},
    "golden_v2.jsonl": {"top": ["expected_answer"], "spans": "gold_spans"},
    "labelset_v1.jsonl": {"top": ["answer"], "spans": None},
    "answers_v1.jsonl": {"top": ["answer"], "spans": None},
    "answers_v2.jsonl": {"top": ["answer"], "spans": None},
}

SANITISED_MARKER = "__text_stripped__"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def measure(path: Path, spec: dict) -> int:
    """Characters of corpus-derived text in a file."""
    if not path.exists():
        return 0
    total = 0
    for row in load_jsonl(path):
        for field in spec["top"]:
            total += len(row.get(field) or "")
        if spec["spans"]:
            for span in row.get(spec["spans"], []):
                total += len(span.get("text") or "")
    return total


def sanitise_file(path: Path, spec: dict) -> tuple[int, int]:
    """Remove text fields, keeping offsets. Returns (rows, characters removed)."""
    rows = load_jsonl(path)
    removed = 0

    for row in rows:
        for field in spec["top"]:
            if row.get(field):
                removed += len(row[field])
                row[field] = None
        if spec["spans"]:
            for span in row.get(spec["spans"], []):
                if span.get("text"):
                    removed += len(span["text"])
                    span.pop("text", None)
        row[SANITISED_MARKER] = True

    write_jsonl(path, rows)
    return len(rows), removed


def reconstruct_file(path: Path, spec: dict, source: LegalBenchRAG) -> int:
    """Restore text from the corpus, for local use.

    Only span text can be restored — it is defined by offsets. Generated answers
    cannot be reconstructed, because they were produced by a model rather than read
    from a document; they live in the LLM cache instead.
    """
    if not spec["spans"]:
        return 0

    rows = load_jsonl(path)
    restored = 0

    for row in rows:
        spans = row.get(spec["spans"], [])
        texts = []
        for span in spans:
            try:
                text = source.document_text(span["doc_id"])[span["start"] : span["end"]]
            except (FileNotFoundError, KeyError):
                continue
            span["text"] = text
            texts.append(text.strip())
            restored += len(text)
        if texts and "expected_answer" in spec["top"]:
            row["expected_answer"] = " ".join(texts)
        row.pop(SANITISED_MARKER, None)

    write_jsonl(path, rows)
    return restored


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="report how much corpus text is present")
    group.add_argument("--sanitise", action="store_true",
                       help="strip text fields, keeping offsets")
    group.add_argument("--reconstruct", action="store_true",
                       help="restore span text from the corpus for local use")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    # --- check ---------------------------------------------------------------
    if args.check:
        print("corpus-derived text in the evaluation files\n")
        total = 0
        for name, spec in TEXT_FIELDS.items():
            path = EVALS / name
            if not path.exists():
                continue
            chars = measure(path, spec)
            total += chars
            state = "clean" if chars == 0 else f"{chars:,} chars"
            print(f"  {name:<24}{state:>18}")

        print(f"\n  total: {total:,} characters")
        if total:
            print("\n  This is verbatim contract text. Phase 1 established that the")
            print("  corpus cannot be redistributed — the underlying datasets each")
            print("  carry their own usage policy and redistribution rights were not")
            print("  confirmed.")
            print("\n  Run --sanitise before making the repository public. Nothing is")
            print("  lost: every span is stored as (doc_id, start, end), which is how")
            print("  the harness reads them anyway.")
        else:
            print("\n  No corpus text present. Safe to publish.")
        return 0

    # --- sanitise ------------------------------------------------------------
    if args.sanitise:
        print("stripping corpus text, keeping offsets\n")
        total_rows = total_removed = 0

        for name, spec in TEXT_FIELDS.items():
            path = EVALS / name
            if not path.exists():
                continue

            if not args.no_backup:
                backup = path.with_suffix(path.suffix + ".full")
                if not backup.exists():
                    shutil.copy2(path, backup)

            rows, removed = sanitise_file(path, spec)
            total_rows += rows
            total_removed += removed
            print(f"  {name:<24}{rows:>4} rows{removed:>12,} chars removed")

        print(f"\n  {total_removed:,} characters removed across {total_rows} rows")
        if not args.no_backup:
            print("\n  Full copies saved as *.jsonl.full — add these to .gitignore.")
        print("\n  Every span is still stored as (doc_id, start, end). Reconstruct")
        print("  locally with --reconstruct, or let the harness read them directly.")

        gitignore = PROJECT_ROOT / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if "*.jsonl.full" not in content:
                gitignore.write_text(
                    content.rstrip() + "\n\n# local copies with corpus text\n*.jsonl.full\n",
                    encoding="utf-8",
                )
                print("\n  Added *.jsonl.full to .gitignore.")
        return 0

    # --- reconstruct ---------------------------------------------------------
    if args.reconstruct:
        print("restoring span text from the corpus\n")
        source = LegalBenchRAG(settings.data_dir)
        total = 0

        for name, spec in TEXT_FIELDS.items():
            path = EVALS / name
            if not path.exists() or not spec["spans"]:
                continue
            restored = reconstruct_file(path, spec, source)
            total += restored
            print(f"  {name:<24}{restored:>12,} chars restored")

        print(f"\n  {total:,} characters restored from the corpus")
        print("\n  Generated answers cannot be reconstructed this way — they came")
        print("  from a model, not a document. They are in the LLM cache; if that")
        print("  was cleared, re-run scripts/phase6_generate.py (cached calls are")
        print("  free, so only genuinely missing answers cost tokens).")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
