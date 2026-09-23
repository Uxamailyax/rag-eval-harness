"""Known-answer tests for the retrieval metrics.

Every expected value below was computed by hand before the code was run. That is
the point of the phase: a wrong number from a buggy metric is indistinguishable
from a wrong number from a bad retriever, so the metrics are proved on cases whose
answers are known independently, before any real retrieval exists.

Each test states its arithmetic in the docstring. If a test fails, the docstring
says what the answer should be and why, so the disagreement is resolvable without
re-deriving it.

Run with:  uv run pytest tests/test_retrieval_metrics.py -v
"""

from __future__ import annotations

import math

import pytest

from rag_eval_harness.metrics.retrieval import (
    CharRange,
    aggregate,
    covered_chars,
    dcg,
    evaluate_query,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_grade,
    total_length,
)

D = "doc1"
E = "doc2"


def r(start: int, end: int, doc: str = D) -> CharRange:
    return CharRange(doc_id=doc, start=start, end=end)


# ---------------------------------------------------------------------------
# Overlap and union arithmetic
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_no_overlap(self):
        """[0,10) and [20,30) share nothing."""
        assert r(0, 10).overlap(r(20, 30)) == 0

    def test_touching_ranges_do_not_overlap(self):
        """Half-open: [0,10) ends before index 10, [10,20) starts at it."""
        assert r(0, 10).overlap(r(10, 20)) == 0

    def test_partial_overlap(self):
        """[0,10) and [5,15) share indices 5..9 = 5 characters."""
        assert r(0, 10).overlap(r(5, 15)) == 5

    def test_contained(self):
        """[0,100) fully contains [30,40) = 10 characters."""
        assert r(0, 100).overlap(r(30, 40)) == 10

    def test_different_documents_never_overlap(self):
        """Identical offsets in different documents are unrelated."""
        assert r(0, 10).overlap(r(0, 10, doc=E)) == 0


class TestTotalLength:
    def test_single(self):
        assert total_length([r(0, 10)]) == 10

    def test_disjoint_sums(self):
        """10 + 10 = 20."""
        assert total_length([r(0, 10), r(20, 30)]) == 20

    def test_overlapping_counted_once(self):
        """[0,10) and [5,15) union to [0,15) = 15, not 20.

        Double-counting here would inflate the precision denominator and make an
        overlapping retriever look worse than it is.
        """
        assert total_length([r(0, 10), r(5, 15)]) == 15

    def test_contained_counted_once(self):
        """[0,100) plus [30,40) is still 100."""
        assert total_length([r(0, 100), r(30, 40)]) == 100

    def test_across_documents_sums(self):
        """Ranges in different documents cannot overlap: 10 + 10 = 20."""
        assert total_length([r(0, 10), r(0, 10, doc=E)]) == 20

    def test_empty(self):
        assert total_length([]) == 0


class TestCoveredChars:
    def test_full_coverage(self):
        """Gold [10,20) is 10 chars, fully inside retrieved [0,100)."""
        assert covered_chars([r(10, 20)], [r(0, 100)]) == 10

    def test_half_coverage(self):
        """Gold [10,20), retrieved [15,25): shared indices 15..19 = 5."""
        assert covered_chars([r(10, 20)], [r(15, 25)]) == 5

    def test_gold_split_across_two_chunks(self):
        """Gold [10,20). Retrieved [10,15) and [15,20) → 5 + 5 = 10.

        A gold span straddling a chunk boundary must still score as fully found.
        """
        assert covered_chars([r(10, 20)], [r(10, 15), r(15, 20)]) == 10

    def test_overlapping_chunks_do_not_double_count(self):
        """Gold [10,20) = 10 chars. Retrieved [10,18) and [12,20) overlap on
        12..17, but coverage is capped at the gold length: 10, not 16."""
        assert covered_chars([r(10, 20)], [r(10, 18), r(12, 20)]) == 10

    def test_multiple_gold_spans_sum(self):
        """Gold [0,10) and [50,60), both fully retrieved: 10 + 10 = 20."""
        assert covered_chars([r(0, 10), r(50, 60)], [r(0, 100)]) == 20

    def test_wrong_document_covers_nothing(self):
        assert covered_chars([r(10, 20)], [r(0, 100, doc=E)]) == 0


# ---------------------------------------------------------------------------
# Relevance grading
# ---------------------------------------------------------------------------


class TestRelevanceGrade:
    def test_full_cover_is_grade_2(self):
        """Chunk [0,100) covers 100% of gold [10,20) ≥ 0.8 → grade 2."""
        assert relevance_grade(r(0, 100), [r(10, 20)]) == 2

    def test_exactly_at_threshold_is_grade_2(self):
        """Gold [0,10) = 10 chars. Chunk [0,8) covers 8 → 0.8, ≥ threshold → 2."""
        assert relevance_grade(r(0, 8), [r(0, 10)]) == 2

    def test_just_below_threshold_is_grade_1(self):
        """Chunk [0,7) covers 7/10 = 0.7 < 0.8 → grade 1."""
        assert relevance_grade(r(0, 7), [r(0, 10)]) == 1

    def test_no_overlap_is_grade_0(self):
        assert relevance_grade(r(50, 60), [r(0, 10)]) == 0

    def test_takes_best_across_gold_spans(self):
        """Chunk covers 30% of one gold span and 100% of another → 2."""
        chunk = r(0, 10)
        gold = [r(0, 30), r(0, 10)]
        assert relevance_grade(chunk, gold) == 2


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit_in_first_position(self):
        assert hit_rate_at_k([r(0, 10)], [r(0, 10)], k=5) == 1.0

    def test_miss(self):
        assert hit_rate_at_k([r(50, 60)], [r(0, 10)], k=5) == 0.0

    def test_single_character_touch_counts_as_hit(self):
        """Chunk [9,50) overlaps gold [0,10) on index 9 only — one character.

        Hit rate says 1.0 anyway. This is exactly why hit rate is a floor and not
        a headline metric.
        """
        assert hit_rate_at_k([r(9, 50)], [r(0, 10)], k=5) == 1.0

    def test_hit_beyond_k_does_not_count(self):
        """Relevant chunk is at position 3; K=2 truncates before it."""
        retrieved = [r(50, 60), r(70, 80), r(0, 10)]
        assert hit_rate_at_k(retrieved, [r(0, 10)], k=2) == 0.0

    def test_empty_retrieval(self):
        assert hit_rate_at_k([], [r(0, 10)], k=5) == 0.0


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_perfect(self):
        """Gold [0,10) entirely inside retrieved [0,10) → 10/10 = 1.0."""
        assert recall_at_k([r(0, 10)], [r(0, 10)], k=5) == 1.0

    def test_half(self):
        """Gold [0,10) = 10 chars, retrieved [0,5) covers 5 → 5/10 = 0.5."""
        assert recall_at_k([r(0, 5)], [r(0, 10)], k=5) == 0.5

    def test_zero(self):
        assert recall_at_k([r(50, 60)], [r(0, 10)], k=5) == 0.0

    def test_two_gold_spans_one_found(self):
        """Gold [0,10) and [50,60) = 20 chars total. Only the first retrieved
        → 10/20 = 0.5."""
        assert recall_at_k([r(0, 10)], [r(0, 10), r(50, 60)], k=5) == 0.5

    def test_gold_split_across_chunks_still_full(self):
        """Gold [10,20). Chunks [10,15) and [15,20) → 10/10 = 1.0."""
        assert recall_at_k([r(10, 15), r(15, 20)], [r(10, 20)], k=5) == 1.0

    def test_returning_everything_scores_perfect(self):
        """Retrieving [0,10000) gets recall 1.0 for a 10-char gold span.

        This is the failure mode that makes recall meaningless alone. The same
        case scores 0.001 on precision — see TestPrecision.
        """
        assert recall_at_k([r(0, 10000)], [r(0, 10)], k=5) == 1.0

    def test_truncated_at_k(self):
        """Gold [0,20) = 20 chars. Chunks: [0,10) at rank 1, [10,20) at rank 2,
        but K=1 keeps only the first → 10/20 = 0.5."""
        assert recall_at_k([r(0, 10), r(10, 20)], [r(0, 20)], k=1) == 0.5


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_perfect(self):
        """Retrieved [0,10) is entirely gold → 10/10 = 1.0."""
        assert precision_at_k([r(0, 10)], [r(0, 10)], k=5) == 1.0

    def test_half(self):
        """Retrieved [0,20) = 20 chars, of which [0,10) is gold → 10/20 = 0.5."""
        assert precision_at_k([r(0, 20)], [r(0, 10)], k=5) == 0.5

    def test_returning_everything_is_punished(self):
        """Retrieved 10,000 chars for a 10-char gold span → 10/10000 = 0.001.

        The counterpart to test_returning_everything_scores_perfect. Precision is
        why large chunks cannot win by sweeping the corpus.
        """
        assert precision_at_k([r(0, 10000)], [r(0, 10)], k=5) == pytest.approx(0.001)

    def test_chunk_size_effect_is_visible(self):
        """Same gold span [0,100), two chunk sizes.

        Small chunk [0,128): recall 100/100 = 1.0, precision 100/128 ≈ 0.781
        Large chunk [0,512): recall 100/100 = 1.0, precision 100/512 ≈ 0.195

        Recall cannot distinguish them; precision can. This is the entire reason
        both are reported in Phase 5.
        """
        gold = [r(0, 100)]
        assert recall_at_k([r(0, 128)], gold, k=5) == 1.0
        assert recall_at_k([r(0, 512)], gold, k=5) == 1.0
        assert precision_at_k([r(0, 128)], gold, k=5) == pytest.approx(100 / 128)
        assert precision_at_k([r(0, 512)], gold, k=5) == pytest.approx(100 / 512)

    def test_empty_retrieval(self):
        assert precision_at_k([], [r(0, 10)], k=5) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_position(self):
        """Relevant at rank 1 → 1/1 = 1.0."""
        assert mrr_at_k([r(0, 10), r(50, 60)], [r(0, 10)], k=5) == 1.0

    def test_second_position(self):
        """Relevant at rank 2 → 1/2 = 0.5."""
        assert mrr_at_k([r(50, 60), r(0, 10)], [r(0, 10)], k=5) == 0.5

    def test_third_position(self):
        """Relevant at rank 3 → 1/3 ≈ 0.3333."""
        assert mrr_at_k(
            [r(50, 60), r(70, 80), r(0, 10)], [r(0, 10)], k=5
        ) == pytest.approx(1 / 3)

    def test_only_first_hit_matters(self):
        """Two systems, same first hit at rank 1, different totals after it.

        Both score 1.0. MRR is blind to how much was found — that is recall's job.
        """
        gold = [r(0, 10), r(50, 60)]
        one_hit = [r(0, 10), r(200, 210), r(300, 310)]
        both_hits = [r(0, 10), r(50, 60), r(300, 310)]
        assert mrr_at_k(one_hit, gold, k=5) == 1.0
        assert mrr_at_k(both_hits, gold, k=5) == 1.0

    def test_no_hit(self):
        assert mrr_at_k([r(50, 60)], [r(0, 10)], k=5) == 0.0

    def test_hit_beyond_k(self):
        """Relevant chunk is at rank 3, K=2 → 0.0."""
        retrieved = [r(50, 60), r(70, 80), r(0, 10)]
        assert mrr_at_k(retrieved, [r(0, 10)], k=2) == 0.0


# ---------------------------------------------------------------------------
# DCG and NDCG
# ---------------------------------------------------------------------------


class TestDCG:
    def test_single_grade_2(self):
        """(2^2 - 1)/log2(2) = 3/1 = 3.0."""
        assert dcg([2]) == pytest.approx(3.0)

    def test_single_grade_1(self):
        """(2^1 - 1)/log2(2) = 1/1 = 1.0."""
        assert dcg([1]) == pytest.approx(1.0)

    def test_grade_0_contributes_nothing(self):
        assert dcg([0, 0, 0]) == 0.0

    def test_two_positions(self):
        """[2, 1] = 3/log2(2) + 1/log2(3) = 3.0 + 0.6309 = 3.6309."""
        expected = 3 / math.log2(2) + 1 / math.log2(3)
        assert dcg([2, 1]) == pytest.approx(expected)

    def test_position_discount_is_real(self):
        """The same grades ordered better score higher: [2,1] > [1,2]."""
        assert dcg([2, 1]) > dcg([1, 2])


class TestNDCG:
    def test_perfect_ordering(self):
        """Grades [2, 1] are already ideal → DCG/IDCG = 1.0."""
        gold = [r(0, 10), r(100, 200)]
        retrieved = [r(0, 10), r(100, 120)]  # grade 2 then grade 1
        assert relevance_grade(retrieved[0], gold) == 2
        assert relevance_grade(retrieved[1], gold) == 1
        assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(1.0)

    def test_reversed_ordering(self):
        """Grades [1, 2] against ideal [2, 1].

        DCG  = 1/log2(2) + 3/log2(3) = 1.0000 + 1.8928 = 2.8928
        IDCG = 3/log2(2) + 1/log2(3) = 3.0000 + 0.6309 = 3.6309
        NDCG = 2.8928 / 3.6309 = 0.7967
        """
        gold = [r(0, 10), r(100, 200)]
        retrieved = [r(100, 120), r(0, 10)]  # grade 1 then grade 2
        expected = (1 / math.log2(2) + 3 / math.log2(3)) / (
            3 / math.log2(2) + 1 / math.log2(3)
        )
        assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(expected)
        assert expected == pytest.approx(0.7967, abs=1e-4)

    def test_all_irrelevant(self):
        """No grades above 0 → IDCG is 0 → defined as 0.0, not a division error."""
        assert ndcg_at_k([r(500, 600)], [r(0, 10)], k=5) == 0.0

    def test_empty_retrieval(self):
        assert ndcg_at_k([], [r(0, 10)], k=5) == 0.0

    def test_ndcg_ignores_what_was_missed(self):
        """One chunk, grade 2, perfectly ordered → NDCG 1.0, while recall is 0.5.

        Gold is [0,10) and [500,510) = 20 chars; only the first is retrieved.
        NDCG measures ordering quality *given what was found*, which is why it is
        never reported without recall.
        """
        gold = [r(0, 10), r(500, 510)]
        retrieved = [r(0, 10)]
        assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(1.0)
        assert recall_at_k(retrieved, gold, k=5) == 0.5


# ---------------------------------------------------------------------------
# Cross-check against sklearn
# ---------------------------------------------------------------------------


class TestAgainstSklearn:
    def test_ndcg_matches_sklearn_on_binary_relevance(self):
        """Independent confirmation of the NDCG formula.

        sklearn.ndcg_score uses the linear gain form (grade / log2(rank+1)); ours
        uses exponential ((2^grade - 1) / log2(rank+1)). The two agree exactly when
        all grades are 0 or 1, because 2^1 - 1 = 1 and 2^0 - 1 = 0.

        This is a genuine external check on the position discount and the
        normalisation, using the case where the two definitions coincide.
        """
        from sklearn.metrics import ndcg_score
        import numpy as np

        grades = [1, 0, 1, 0, 0]
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]  # already in rank order

        sklearn_value = ndcg_score(np.array([grades]), np.array([scores]))
        ours = dcg(grades) / dcg(sorted(grades, reverse=True))
        assert ours == pytest.approx(sklearn_value)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestEvaluateAndAggregate:
    def test_evaluate_query_returns_all_metrics(self):
        result = evaluate_query([r(0, 10)], [r(0, 10)], ks=(5, 10))
        for k in (5, 10):
            for name in ("hit_rate", "recall", "precision", "mrr", "ndcg"):
                assert f"{name}@{k}" in result

    def test_macro_average(self):
        """Two queries scoring 1.0 and 0.0 average to 0.5."""
        per_query = [{"recall@5": 1.0}, {"recall@5": 0.0}]
        assert aggregate(per_query)["recall@5"] == 0.5

    def test_macro_ignores_gold_span_size(self):
        """A query with a 5,000-char gold span counts the same as one with 100.

        Macro-averaging reports what the average *query* experiences. Micro would
        let a handful of long spans dominate the headline number.
        """
        long_gold = [r(0, 5000)]
        short_gold = [r(0, 100)]
        q1 = evaluate_query([r(0, 5000)], long_gold, ks=(5,))  # recall 1.0
        q2 = evaluate_query([], short_gold, ks=(5,))  # recall 0.0
        assert aggregate([q1, q2])["recall@5"] == 0.5

    def test_empty_aggregate(self):
        assert aggregate([]) == {}


# ---------------------------------------------------------------------------
# The headline case: why character-level scoring exists
# ---------------------------------------------------------------------------


def test_chunk_level_scoring_would_favour_large_chunks():
    """The measurement artifact this whole module exists to avoid.

    Gold span [1000, 1100), 100 characters. Two chunking configurations, each
    returning the single chunk that contains it:

        128-char chunks → chunk [1000, 1128)
        512-char chunks → chunk [1000, 1512)

    Chunk-level scoring asks only "did a returned chunk contain gold?" — both
    score 1.0, and the ablation concludes the two are equivalent.

    Character-level scoring sees that the larger chunk delivered 412 irrelevant
    characters to get the same 100 gold ones:

        recall     both 1.0
        precision  100/128 ≈ 0.781  vs  100/512 ≈ 0.195

    That gap is the finding. Without precision it is invisible.
    """
    gold = [r(1000, 1100)]
    small = [r(1000, 1128)]
    large = [r(1000, 1512)]

    assert recall_at_k(small, gold, k=5) == recall_at_k(large, gold, k=5) == 1.0
    assert precision_at_k(small, gold, k=5) == pytest.approx(100 / 128)
    assert precision_at_k(large, gold, k=5) == pytest.approx(100 / 512)
    assert precision_at_k(small, gold, k=5) > precision_at_k(large, gold, k=5)
