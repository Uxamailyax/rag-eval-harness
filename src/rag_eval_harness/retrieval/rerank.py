"""Cross-encoder reranking: the second stage.

Dense retrieval embeds the query and each chunk **separately**, then compares the
two vectors. The chunk was encoded without any knowledge of the question — the
same vector serves every query ever asked. That independence is what makes it
fast: encode the corpus once, and each search is a matrix multiply.

A cross-encoder does the opposite. It reads the query and one chunk **together**
in a single forward pass and outputs a relevance score. Nothing is precomputed and
nothing is reusable, because the score only exists for that exact pair.

That makes it far more accurate and far too slow to run over a corpus. One query
against 64,120 chunks would be 64,120 forward passes. So it runs as a second stage:
cheap wide retrieval finds ~50 candidates, the expensive model reorders those 50.

**Why this might fail here, and why that would be the more interesting result.**
The LegalBench-RAG authors reported that a general-purpose reranker *degraded*
retrieval on legal text. The mechanism is not that legal language is "hard" — it is
that a general model does not know which words carry the legal weight. In a
contract, "Receiving Party" is not jargon for an ordinary idea; it is a precise
reference to a party defined on page one. A general model reads it as two ordinary
English words, so it cannot distinguish a binding obligation from background prose
that happens to use similar vocabulary.

Lacking that signal, it falls back on what it was trained to reward: passages that
read like clear, well-formed answers. In contracts those are definitions and
recitals — not the operative clause buried mid-section. So the correct chunk sitting
at rank 4 can be pushed to rank 12 by something that merely looks more like an
answer, and recall@10 drops.

Phase 4 established that the gold chunk is often present but ranked low (MRR 0.16,
so roughly rank 6 when found), which is exactly the condition reranking exists to
fix. Whether it does is the measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "retrieval.rerank needs `sentence-transformers`. "
        "Install with: uv add sentence-transformers"
    ) from exc

from rag_eval_harness.retrieval.bm25 import RetrievedChunk

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class RerankTiming:
    """Latency split across the two stages.

    Reported separately because the shipping decision depends on the split, not
    the total: a reranker that adds 30 ms to a 500 ms pipeline is different from
    one that adds 300 ms to a 20 ms pipeline, even though both "add latency".
    """

    first_stage_ms: float
    rerank_ms: float
    n_candidates: int

    @property
    def total_ms(self) -> float:
        return self.first_stage_ms + self.rerank_ms

    @property
    def overhead_ratio(self) -> float:
        if self.first_stage_ms == 0:
            return float("inf")
        return self.rerank_ms / self.first_stage_ms


class CrossEncoderReranker:
    """Reorders candidates by joint query-chunk scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.device = str(getattr(self._model, "device", device or "cpu"))

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 10
    ) -> tuple[list[RetrievedChunk], float]:
        """Rescore `candidates` and return the top K, plus elapsed milliseconds.

        Ranks are renumbered from 1 after reordering, so downstream metrics see a
        clean ranking rather than the first-stage positions.
        """
        if not candidates:
            return [], 0.0

        pairs = [(query, c.chunk.text) for c in candidates]

        started = time.perf_counter()
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)

        reranked = [
            RetrievedChunk(
                chunk=candidates[i].chunk, score=float(scores[i]), rank=new_rank
            )
            for new_rank, i in enumerate(order[:top_k], start=1)
        ]
        return reranked, elapsed_ms


class TwoStageRetriever:
    """First-stage retrieval followed by cross-encoder reranking.

    `candidate_k` is the reranker's ceiling as well as its cost: a gold chunk
    ranked 60th by the first stage can never be recovered if only 50 candidates
    are passed through, no matter how good the reranker is. Larger values raise
    that ceiling and the latency together, linearly.
    """

    def __init__(
        self,
        base_retriever: object,
        reranker: CrossEncoderReranker,
        candidate_k: int = 50,
    ):
        self.base = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.last_timing: RerankTiming | None = None

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        started = time.perf_counter()
        candidates = self.base.retrieve(query, top_k=self.candidate_k)  # type: ignore[attr-defined]
        first_stage_ms = (time.perf_counter() - started) * 1000.0

        reranked, rerank_ms = self.reranker.rerank(query, candidates, top_k=top_k)

        self.last_timing = RerankTiming(
            first_stage_ms=first_stage_ms,
            rerank_ms=rerank_ms,
            n_candidates=len(candidates),
        )
        return reranked


def rank_movement(
    before: list[RetrievedChunk], after: list[RetrievedChunk]
) -> dict[str, int]:
    """How much the reranker rearranged things.

    A reranker that changes nothing is not earning its latency; one that rewrites
    the order completely is making a strong claim. Either way the aggregate number
    is worth reporting alongside the quality delta, because it separates "the
    reranker helped" from "the reranker barely ran".
    """
    before_keys = {
        (c.chunk.doc_id, c.chunk.start): c.rank for c in before
    }
    promoted = demoted = unchanged = new_entries = 0

    for hit in after:
        key = (hit.chunk.doc_id, hit.chunk.start)
        old = before_keys.get(key)
        if old is None:
            new_entries += 1
        elif hit.rank < old:
            promoted += 1
        elif hit.rank > old:
            demoted += 1
        else:
            unchanged += 1

    return {
        "promoted": promoted,
        "demoted": demoted,
        "unchanged": unchanged,
        "new_in_top_k": new_entries,
    }
