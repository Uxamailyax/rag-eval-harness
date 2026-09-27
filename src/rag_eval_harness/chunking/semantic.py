"""Semantic chunking: cut where the topic shifts.

The other three chunkers cut at positions decided by counting — tokens,
paragraphs, sentences. This one reads the text. It embeds every sentence, measures
how similar each sentence is to the one before it, and cuts where that similarity
drops: the points where the document stops talking about one thing and starts
talking about another.

The appeal is obvious. A chunk that contains one coherent topic should retrieve
better than a chunk that happens to hold the end of one clause and the start of
another.

The cost is also obvious: every sentence in the corpus has to be embedded before
any chunk exists. That is roughly the same work as embedding the chunks themselves,
paid twice.

**Published results disagree on whether it earns that cost**, and they disagree
for an instructive reason: two benchmarks reached opposite conclusions because one
measured retrieval recall and the other measured end-to-end answer accuracy.
Semantic chunking can improve the first while degrading the second. Reproducing
that contradiction would be a stronger result than either finding alone.

On contracts specifically there is reason for doubt. Legal drafting is already
segmented by numbered clauses, so the topic boundaries the model is looking for
may be exactly the paragraph breaks the trivial chunker already uses — in which
case this is an expensive way to get the same cuts.
"""

from __future__ import annotations

import re

import numpy as np

from rag_eval_harness.chunking.fixed import Chunk, DEFAULT_TOKENIZER
from rag_eval_harness.chunking.structural import _TokenCounter, _tile

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "chunking.semantic needs `sentence-transformers`. "
        "Install with: uv add sentence-transformers"
    ) from exc

# Sentence boundary. Deliberately simple: legal text is full of abbreviations
# ("Inc.", "No.", "Sec.") that a naive rule mis-splits, but the chunker only needs
# *candidate* cut points, and a few spurious ones cost nothing because the
# similarity test decides which are used.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Fraction of candidate boundaries to cut at. 0.25 means the 25% of boundaries
# with the lowest similarity become chunk edges — a percentile threshold rather
# than an absolute one, because raw cosine values vary by document.
DEFAULT_BREAKPOINT_PERCENTILE = 25


class SemanticChunker:
    """Chunk at points where consecutive sentences are least similar."""

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        model_name: str = DEFAULT_TOKENIZER,
        breakpoint_percentile: int = DEFAULT_BREAKPOINT_PERCENTILE,
        batch_size: int = 128,
        model: SentenceTransformer | None = None,
        counter: _TokenCounter | None = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.breakpoint_percentile = breakpoint_percentile
        self.batch_size = batch_size
        self._model = model or SentenceTransformer(model_name)
        self._counter = counter or _TokenCounter(tokenizer_name)

    @property
    def name(self) -> str:
        return "semantic"

    # --- sentence splitting ------------------------------------------------

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        """Sentence spans as (start, end) character offsets, tiling the text."""
        bounds = [0]
        for match in SENTENCE_END.finditer(text):
            bounds.append(match.end())
        bounds.append(len(text))
        return [(s, e) for s, e in zip(bounds, bounds[1:]) if e > s]

    # --- chunking ----------------------------------------------------------

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        sentences = self._sentences(text)
        if len(sentences) < 3:
            return _tile(doc_id, text, [])

        bodies = [text[s:e] for s, e in sentences]
        vectors = self._model.encode(
            bodies,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Similarity between each adjacent pair. Normalised vectors, so the dot
        # product is the cosine. A low value means the document changed subject
        # between those two sentences.
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)

        # Percentile rather than a fixed cutoff: absolute cosine values differ
        # between a dense merger agreement and a short privacy policy, so a fixed
        # threshold would cut one to pieces and leave the other whole.
        threshold = float(np.percentile(similarities, self.breakpoint_percentile))
        candidates = [
            sentences[i + 1][0] for i, sim in enumerate(similarities) if sim <= threshold
        ]

        cuts = self._enforce_budget(text, sorted(set(candidates)))
        return _tile(doc_id, text, cuts)

    def _enforce_budget(self, text: str, candidates: list[int]) -> list[int]:
        """Keep semantic cuts, but add more when a segment exceeds the budget.

        Topic boundaries alone give no size control — a document can discuss one
        subject for 4,000 tokens. Without this the resulting chunks would vary so
        wildly that comparing this strategy against the fixed-size one would be
        comparing two different things at once.

        Oversized segments are split at their own least-similar internal points
        where possible, falling back to sentence boundaries.
        """
        cuts: list[int] = []
        bounds = [0] + candidates + [len(text)]

        for start, end in zip(bounds, bounds[1:]):
            if end <= start:
                continue
            if start > 0:
                cuts.append(start)

            segment = text[start:end]
            if self._counter.count(segment) <= self.chunk_size_tokens:
                continue

            # Over budget: subdivide at sentence ends, greedily filling.
            sub = self._sentences(segment)
            group_start = 0
            group_end = 0
            for s_start, s_end in sub:
                candidate = segment[group_start:s_end]
                if (
                    group_end > group_start
                    and self._counter.count(candidate) > self.chunk_size_tokens
                ):
                    cuts.append(start + group_end)
                    group_start = group_end
                group_end = s_end

        return sorted(set(c for c in cuts if 0 < c < len(text)))
