"""Display a labelling case with its full excerpt text.

The labelling set stores which chunks were retrieved, not their text — the text lives
in the corpus and is reconstructed from character offsets. This prints a case as the
judge will see it, so the human and the model are answering the same question about
the same material.

Run with:  uv run python scripts/phase7_show_case.py case:00
           uv run python scripts/phase7_show_case.py case:00 --full
           uv run python scripts/phase7_show_case.py --all > cases.txt
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_eval_harness.config import settings
from rag_eval_harness.ground_truth.legalbench import LegalBenchRAG

LABELSET_PATH = settings.data_dir.parent / "evals" / "labelset_v1.jsonl"

# How much of each excerpt to show by default. Legal chunks run to ~2,500
# characters, and thirty cases at ten excerpts each is a great deal of reading.
PREVIEW_CHARS = 700


def load_cases() -> list[dict]:
    if not LABELSET_PATH.exists():
        raise FileNotFoundError(
            f"missing {LABELSET_PATH}\n"
            "Run: uv run python scripts/phase7_build_labelset.py"
        )
    with open(LABELSET_PATH, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def show(case: dict, source: LegalBenchRAG, limit: int | None) -> None:
    print("=" * 88)
    print(f"[{case['case_id']}]")
    print("=" * 88)

    print(f"\nQUESTION")
    print(f"  {case['question']}")

    print(f"\nANSWER")
    for line in case["answer"].split("\n"):
        if line.strip():
            print(f"  {line.strip()}")

    print(f"\nEXCERPTS PROVIDED ({len(case['retrieved'])})")
    print("-" * 88)

    for hit in case["retrieved"]:
        name = hit["doc_id"].split("/")[-1]
        print(f"\n  [Excerpt {hit['rank']} — {name}]")
        try:
            text = source.document_text(hit["doc_id"])[hit["start"] : hit["end"]]
        except FileNotFoundError:
            print("    (document unavailable)")
            continue
        body = " ".join(text.split())
        if limit and len(body) > limit:
            body = body[:limit] + " ..."
        # Indent so the excerpt is visually separate from the answer above.
        for i in range(0, len(body), 84):
            print(f"    {body[i : i + 84]}")

    print(f"\n{'-' * 88}")
    print("  Is every claim in the answer supported by these excerpts?")
    print("  1 = faithful   0 = unfaithful")
    print("  (a refusal asserts nothing unsupported, so it counts as faithful)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", nargs="?", help="e.g. case:00")
    parser.add_argument("--all", action="store_true", help="print every case")
    parser.add_argument("--full", action="store_true",
                        help="show complete excerpts rather than previews")
    args = parser.parse_args()

    cases = load_cases()
    source = LegalBenchRAG(settings.data_dir)
    limit = None if args.full else PREVIEW_CHARS

    if args.all:
        for case in cases:
            show(case, source, limit)
            print("\n")
        return 0

    if not args.case_id:
        print(f"{len(cases)} cases available:\n")
        for case in cases:
            preview = " ".join(case["answer"].split())[:58]
            print(f"  {case['case_id']}  {preview}")
        print(f"\nShow one:  uv run python scripts/phase7_show_case.py case:00")
        print(f"Show all:  uv run python scripts/phase7_show_case.py --all > cases.txt")
        return 0

    match = [c for c in cases if c["case_id"] == args.case_id]
    if not match:
        print(f"no case {args.case_id}")
        return 1

    show(match[0], source, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
