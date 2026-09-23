"""Character-level retrieval metrics.

Every metric here scores over *characters*, not chunks. That is the single most
important design decision in the harness, and it exists because chunk-level scoring
is not comparable across chunking configurations.

The problem it solves: a 512-token chunk is mechanically more likely to overlap a
gold span than a 128-token chunk. Score at chunk level and Recall@K rises with chunk
size as a pure artifact of the window being wider — the ablation in Phase 5 would
conclude "bigger chunks are better" while measuring nothing about retrieval quality.

Scoring over characters removes that. A retriever is credited for the gold characters
it actually returned, and charged for every character it returned that was not gold.
Precision is the term that punishes large chunks, which is why recall is never
reported without it.

All five metrics are implemented here rather than imported. Implementing them is what
makes them defensible in an interview, and no library scores at character level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharRange:
    """A half-open character range [start, end) within one document."""

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid range [{self.start}, {self.end}) in {self.doc_id}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: "CharRange") -> int:
        """Characters shared with `other`. Zero for different documents."""
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


def total_length(ranges: list[CharRange]) -> int:
    """Total characters covered, counting overlapping ranges once.

    Deduplication matters: if two retrieved chunks overlap, the shared characters
    were returned once from the reader's point of view, and counting them twice
    would inflate the precision denominator.
    """
    return sum(_merge(ranges_for_doc) for ranges_for_doc in _by_doc(ranges).values())


def covered_chars(gold: list[CharRange], retrieved: list[CharRange]) -> int:
    """Gold characters present anywhere in `retrieved`, each counted once."""
    total = 0
    for doc_id, gold_ranges in _by_doc(gold).items():
        retrieved_ranges = _by_doc(retrieved).get(doc_id, [])
        if not retrieved_ranges:
            continue
        for g in gold_ranges:
            marks = [False] * g.length
            for r in retrieved_ranges:
                lo = max(g.start, r.start)
                hi = min(g.end, r.end)
                for i in range(lo - g.start, hi - g.start):
                    marks[i] = True
            total += sum(marks)
    return total


def _by_doc(ranges: list[CharRange]) -> dict[str, list[CharRange]]:
    out: dict[str, list[CharRange]] = {}
    for r in ranges:
        out.setdefault(r.doc_id, []).append(r)
    return out


def _merge(ranges: list[CharRange]) -> int:
    """Length of the union of ranges within a single document."""
    if not ranges:
        return 0
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    total = 0
    cur_start, cur_end = ordered[0].start, ordered[0].end
    for r in ordered[1:]:
        if r.start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = r.start, r.end
        else:
            cur_end = max(cur_end, r.end)
    total += cur_end - cur_start
    return total


# ---------------------------------------------------------------------------
# Relevance grading
# ---------------------------------------------------------------------------

HIGH_RELEVANCE_THRESHOLD = 0.8


def relevance_grade(
    chunk: CharRange,
    gold: list[CharRange],
    threshold: float = HIGH_RELEVANCE_THRESHOLD,
) -> int:
    """Graded relevance for one retrieved chunk: 2, 1, or 0.

    NDCG needs grades, and this corpus supplies only binary spans, so the grades
    are derived from how much of a gold span the chunk covers:

        2  covers >= `threshold` of at least one gold span — substantively answers it
        1  overlaps a gold span but covers less — partially relevant
        0  no overlap

    The derivation is stated in the README because "who assigned these grades" is
    the first question an interviewer asks about a graded metric.
    """
    best = 0
    for g in gold:
        ov = chunk.overlap(g)
        if ov == 0:
            continue
        fraction = ov / g.length if g.length else 0.0
        best = max(best, 2 if fraction >= threshold else 1)
    return best


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def hit_rate_at_k(retrieved: list[CharRange], gold: list[CharRange], k: int) -> float:
    """Did *any* of the top-K chunks touch a gold span? 1.0 or 0.0.

    Ignores ranking and quantity entirely. Useful as a floor — a system failing
    hit rate is not a ranking problem, it is a search problem — and misleading as
    a headline, because touching one character of one gold span scores the same
    as returning all of them.
    """
    top = retrieved[:k]
    return 1.0 if any(relevance_grade(c, gold) > 0 for c in top) else 0.0


def recall_at_k(retrieved: list[CharRange], gold: list[CharRange], k: int) -> float:
    """Fraction of gold characters present in the top-K chunks.

    Reporting this alone is the easiest way to make a retriever look better than
    it is: returning the entire corpus scores 1.0. It is only meaningful next to
    precision.
    """
    gold_total = sum(g.length for g in gold)
    if gold_total == 0:
        return 0.0
    return covered_chars(gold, retrieved[:k]) / gold_total


def precision_at_k(retrieved: list[CharRange], gold: list[CharRange], k: int) -> float:
    """Fraction of returned characters that were gold.

    This is the term that makes chunk sizes comparable. A larger chunk sweeps up
    more gold characters, but also more non-gold ones, and precision charges for
    them. Commonly dropped on the assumption that the LLM will ignore irrelevant
    context — which it does, until an irrelevant chunk contradicts the right answer.
    """
    top = retrieved[:k]
    retrieved_total = total_length(top)
    if retrieved_total == 0:
        return 0.0
    return covered_chars(gold, top) / retrieved_total


def mrr_at_k(retrieved: list[CharRange], gold: list[CharRange], k: int) -> float:
    """Reciprocal rank of the first relevant chunk: 1/rank, or 0 if none in top-K.

    Cares only about the first hit. A system that puts one relevant chunk at
    position 1 and nothing else relevant scores 1.0, identically to a system that
    returns every relevant chunk starting at position 1. That blindness is the
    point: MRR answers "how far must the reader scroll", not "how much was found".
    """
    for idx, chunk in enumerate(retrieved[:k], start=1):
        if relevance_grade(chunk, gold) > 0:
            return 1.0 / idx
    return 0.0


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain, (2^grade - 1) / log2(rank + 1).

    The exponential gain form is used rather than plain grade, so a grade-2 chunk
    is worth meaningfully more than a grade-1 one rather than merely twice as much.
    """
    return sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(retrieved: list[CharRange], gold: list[CharRange], k: int) -> float:
    """NDCG@K over derived relevance grades.

    The one metric that distinguishes a good ranking from a lucky one: it rewards
    putting the most relevant chunks first, with a logarithmic discount by position.

    The ideal ranking is the retrieved chunks' own grades sorted descending. That
    makes this a measure of *ordering quality given what was found* — a retriever
    that found little but ordered it perfectly scores 1.0 here, which is why NDCG
    is never reported without recall.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    grades = [relevance_grade(c, gold) for c in top]
    ideal = sorted(grades, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(grades) / idcg


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def evaluate_query(
    retrieved: list[CharRange],
    gold: list[CharRange],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """All five metrics at each K, for one query."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, gold, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        out[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average across queries: every query counts equally.

    Macro rather than micro because a query with a 5,000-character gold span should
    not outweigh fifty queries with 100-character spans. The harness reports what
    the average *query* experiences, not what the average *character* does.
    """
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
