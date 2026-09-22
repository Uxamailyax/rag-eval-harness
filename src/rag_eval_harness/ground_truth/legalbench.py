"""LegalBench-RAG loader.

Reads the four benchmark JSON files and the corpus text files their spans point
into.

Two reading decisions dominate this file, and Phase 1 established both empirically:

1. `encoding="utf-8"` is explicit. Python's default on Windows is cp1252, which
   decodes the same bytes into a different number of characters and silently
   shifts every offset after the first non-ASCII byte.

2. Newlines are translated, not preserved. The corpus stores CRLF. Reading with
   `newline=""` keeps "\r\n" as two characters, while the annotators' offsets were
   computed against text read with Python's default universal-newline mode, where
   it is one. Preserving CRLF put every span one character out per preceding line
   break — 23% of spans returned the wrong text before this was corrected.

Both failure modes are invisible from the output: the text still reads like a
contract and the spans still return plausible sentences. Only comparing against
the stored answer reveals them.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from rag_eval_harness.ground_truth.base import GroundTruthSource, Query, Span

SUBSET_FILES = {
    "contractnli": "contractnli.json",
    "cuad": "cuad.json",
    "maud": "maud.json",
    "privacy_qa": "privacy_qa.json",
}

# Characters Windows forbids in filenames. Corpus paths containing them cannot
# exist on this filesystem, so those documents are unreachable here regardless of
# how the archive was extracted.
WINDOWS_ILLEGAL = set('<>:"|?*')


def _norm(text: str) -> str:
    """NFC-normalise for filename comparison.

    The same accented character can be stored as one codepoint or as a base plus
    a combining mark. They compare unequal as strings but name the same file.
    """
    return unicodedata.normalize("NFC", text)


class LegalBenchRAG(GroundTruthSource):
    """LegalBench-RAG as a GroundTruthSource.

    `doc_id` is the corpus-relative path exactly as the benchmark JSON writes it,
    e.g. "contractnli/AGProjects-NDA.txt". Keeping the dataset's own identifier
    avoids a translation layer that could introduce its own bugs.
    """

    def __init__(
        self,
        data_dir: Path,
        subsets: list[str] | None = None,
        encoding: str = "utf-8",
    ):
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"
        self.benchmarks_dir = self.data_dir / "benchmarks"
        self.encoding = encoding
        self.subsets = subsets or list(SUBSET_FILES)

        unknown = set(self.subsets) - set(SUBSET_FILES)
        if unknown:
            raise ValueError(f"unknown subset(s): {sorted(unknown)}")
        if not self.corpus_dir.is_dir():
            raise FileNotFoundError(f"corpus not found at {self.corpus_dir}")
        if not self.benchmarks_dir.is_dir():
            raise FileNotFoundError(f"benchmarks not found at {self.benchmarks_dir}")

        self._queries: list[Query] | None = None
        self._text_cache: dict[str, str] = {}
        self._disk_index: dict[str, Path] | None = None

    # --- filename resolution ----------------------------------------------

    def _build_disk_index(self) -> dict[str, Path]:
        """Map normalised corpus-relative path -> actual path on disk.

        Built once. Lets a doc_id whose accents are encoded differently in the
        JSON than on the filesystem still resolve to the right file.
        """
        index: dict[str, Path] = {}
        for path in self.corpus_dir.rglob("*.txt"):
            rel = path.relative_to(self.corpus_dir).as_posix()
            index[_norm(rel)] = path
        return index

    def resolve_path(self, doc_id: str) -> Path | None:
        """Actual file for a doc_id, or None if it is not on disk."""
        if self._disk_index is None:
            self._disk_index = self._build_disk_index()

        direct = self.corpus_dir / doc_id
        if direct.is_file():
            return direct
        return self._disk_index.get(_norm(doc_id))

    @staticmethod
    def is_unreachable_on_windows(doc_id: str) -> bool:
        """True if the path contains characters Windows forbids in filenames."""
        return any(ch in WINDOWS_ILLEGAL for ch in doc_id)

    # --- GroundTruthSource ------------------------------------------------

    def queries(self) -> list[Query]:
        if self._queries is None:
            self._queries = self._load_queries()
        return self._queries

    def document_text(self, doc_id: str) -> str:
        """Decoded text of one corpus file.

        Cached because one document backs many queries, and because every caller
        must see the identical string — otherwise offsets could differ between
        callers.
        """
        if doc_id in self._text_cache:
            return self._text_cache[doc_id]

        path = self.resolve_path(doc_id)
        if path is None:
            raise FileNotFoundError(f"corpus file missing: {doc_id}")

        # newline=None (the default) translates CRLF to LF, matching how the
        # annotators' offsets were computed. See module docstring.
        with open(path, "r", encoding=self.encoding) as fh:
            text = fh.read()

        self._text_cache[doc_id] = text
        return text

    def has_document(self, doc_id: str) -> bool:
        return self.resolve_path(doc_id) is not None

    def document_ids(self) -> list[str]:
        ids: list[str] = []
        for subset in self.subsets:
            subset_dir = self.corpus_dir / subset
            if not subset_dir.is_dir():
                continue
            for path in sorted(subset_dir.rglob("*.txt")):
                ids.append(path.relative_to(self.corpus_dir).as_posix())
        return ids

    # --- loading ----------------------------------------------------------

    def _load_queries(self) -> list[Query]:
        out: list[Query] = []
        for subset in self.subsets:
            path = self.benchmarks_dir / SUBSET_FILES[subset]
            if not path.is_file():
                raise FileNotFoundError(f"benchmark file missing: {path}")

            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            for idx, test in enumerate(payload.get("tests", [])):
                spans = tuple(
                    Span(
                        doc_id=snippet["file_path"],
                        start=int(snippet["span"][0]),
                        end=int(snippet["span"][1]),
                        answer=snippet.get("answer"),
                    )
                    for snippet in test.get("snippets", [])
                )
                if not spans:
                    continue
                out.append(
                    Query(
                        query_id=f"{subset}:{idx:05d}",
                        text=test["query"],
                        spans=spans,
                        subset=subset,
                        tags=tuple(test.get("tags", [])),
                    )
                )
        return out

    # --- convenience ------------------------------------------------------

    def slice_span(self, span: Span) -> str | None:
        """Text at a gold span, or None when the document is unreachable.

        Phase 1 is the check that this equals `span.answer`.
        """
        try:
            return self.document_text(span.doc_id)[span.start : span.end]
        except FileNotFoundError:
            return None

    def usable_queries(self) -> list[Query]:
        """Queries whose every gold span resolves to a file on disk.

        A query with one unreachable span cannot be scored fairly: its recall
        denominator would include characters no retriever could ever return.
        Dropping the whole query is the honest handling.
        """
        return [
            q for q in self.queries() if all(self.has_document(s.doc_id) for s in q.spans)
        ]

    def subset_counts(self, queries: list[Query] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for query in queries if queries is not None else self.queries():
            counts[query.subset] = counts.get(query.subset, 0) + 1
        return counts
