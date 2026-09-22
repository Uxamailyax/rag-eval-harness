"""The ground-truth interface.

Everything downstream — chunkers, retrievers, metrics — talks to a
`GroundTruthSource`, never to a specific dataset. That boundary is what lets the
same harness later measure a different corpus (e.g. labelled Accounts.ai receipts)
without touching any evaluation code.

The contract is deliberately small: a list of queries, the gold spans for each,
and the text of a document. Nothing dataset-specific leaks past it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """A gold answer location: a half-open character range in one document.

    Offsets index into the document's *decoded* text (Python `str`), not bytes.
    `answer` is the text the dataset claims lives at that range; it exists so the
    span can be verified, and is not used for scoring.
    """

    doc_id: str
    start: int
    end: int
    answer: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end}) in {self.doc_id}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Query:
    """One evaluation query and the gold spans that answer it.

    A query can have several spans, possibly across several documents. That is why
    retrieval is scored over characters rather than over whole chunks.
    """

    query_id: str
    text: str
    spans: tuple[Span, ...]
    subset: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def doc_ids(self) -> set[str]:
        return {s.doc_id for s in self.spans}

    @property
    def total_gold_chars(self) -> int:
        """Denominator for character-level recall."""
        return sum(s.length for s in self.spans)


class GroundTruthSource(ABC):
    """A corpus plus its labelled queries."""

    @abstractmethod
    def queries(self) -> list[Query]:
        """Every labelled query in this source."""

    @abstractmethod
    def document_text(self, doc_id: str) -> str:
        """Full decoded text of one document. Must be stable across calls: gold
        spans are offsets into exactly this string."""

    @abstractmethod
    def document_ids(self) -> list[str]:
        """Every document in the corpus, including ones no query touches."""

    def document_count(self) -> int:
        return len(self.document_ids())

    def query_count(self) -> int:
        return len(self.queries())
