"""Hybrid retrieval: Reciprocal Rank Fusion of two ranked lists.

RRF scores each chunk by its *position* in each list rather than by the score it
received:

    score = sum over lists of  1 / (k + rank)

Ranks are used instead of raw scores because the two retrievers' scores are not
comparable — BM25 runs to ~150 on this corpus while cosine similarity runs 0 to 1,
and any normalisation between them is an arbitrary choice that changes results.
A rank of 3 means the same thing in both lists by construction.

The constant k (60 by convention, from the original RRF paper) flattens the curve.
Without it, rank 1 would score 1.0 and rank 2 would score 0.5, so fusion would
collapse into "whatever the stronger retriever put first". At k=60 the gap between
ranks 1 and 2 is under 2%, which makes *agreement between the retrievers* matter
more than either one's confidence.

How it fails: when both retrievers independently rank the same wrong chunk
moderately well, RRF promotes it above chunks that only one retriever found.
Agreement amplifies shared errors as readily as shared correctness, so hybrid can
score below both of its components.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval_harness.retrieval.bm25 import RetrievedChunk

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusionDetail:
    """Where one fused chunk came from. Used to explain hybrid's behaviour."""

    doc_id: str
    start: int
    end: int
    rrf_score: float
    ranks: dict[str, int]  # retriever name -> rank, absent if it did not return it

    @property
    def found_by_both(self) -> bool:
        return len(self.ranks) > 1


def _key(hit: RetrievedChunk) -> tuple[str, int, int]:
    return (hit.chunk.doc_id, hit.chunk.start, hit.chunk.end)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]],
    top_k: int = 10,
    k: int = DEFAULT_RRF_K,
) -> tuple[list[RetrievedChunk], list[FusionDetail]]:
    """Fuse named ranked lists. Returns the fused top-K and per-chunk provenance.

    Chunks are identified by (doc_id, start, end) rather than by object identity,
    so the same span retrieved by both methods fuses correctly.
    """
    scores: dict[tuple[str, int, int], float] = {}
    ranks: dict[tuple[str, int, int], dict[str, int]] = {}
    chunks: dict[tuple[str, int, int], RetrievedChunk] = {}

    for name, hits in ranked_lists.items():
        for hit in hits:
            key = _key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + hit.rank)
            ranks.setdefault(key, {})[name] = hit.rank
            chunks.setdefault(key, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    fused: list[RetrievedChunk] = []
    details: list[FusionDetail] = []
    for rank, (key, score) in enumerate(ordered, start=1):
        source = chunks[key]
        fused.append(RetrievedChunk(chunk=source.chunk, score=score, rank=rank))
        details.append(
            FusionDetail(
                doc_id=key[0],
                start=key[1],
                end=key[2],
                rrf_score=score,
                ranks=dict(ranks[key]),
            )
        )
    return fused, details


class HybridRetriever:
    """BM25 and dense, fused with RRF.

    Each component retrieves `candidate_k` results and the fusion selects `top_k`
    from the union. Retrieving deeper than the final K matters: a chunk ranked 15th
    by one retriever and 3rd by the other should be able to surface, and it cannot
    if each list is truncated at 10 before fusion.
    """

    def __init__(
        self,
        retrievers: dict[str, object],
        candidate_k: int = 50,
        k: int = DEFAULT_RRF_K,
    ):
        if len(retrievers) < 2:
            raise ValueError("hybrid needs at least two retrievers")
        self.retrievers = retrievers
        self.candidate_k = candidate_k
        self.k = k
        self.last_details: list[FusionDetail] = []

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        lists = {
            name: r.retrieve(query, top_k=self.candidate_k)  # type: ignore[attr-defined]
            for name, r in self.retrievers.items()
        }
        fused, details = reciprocal_rank_fusion(lists, top_k=top_k, k=self.k)
        self.last_details = details
        return fused
