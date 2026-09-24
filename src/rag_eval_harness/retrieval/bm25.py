"""BM25 retrieval over chunks.

The lexical baseline: term frequency, inverse document frequency, length
normalisation. Strong on exact terms, defined terms, party names and section
references — all of which legal queries are full of. Blind to synonymy: a query
saying "terminate" will not match a clause saying "rescind".

Indexing is over *chunks*, not whole documents, so every hit carries the character
offsets the metrics need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "retrieval.bm25 needs `rank_bm25`. Install with: uv add rank-bm25"
    ) from exc

from rag_eval_harness.chunking.fixed import Chunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. No stemming, no stopword removal.

    Deliberately plain: BM25's IDF weighting already discounts common words, and
    stemming legal text is a decision with its own trade-offs ("terminating" and
    "termination" mean different things in a clause) that the ablation does not
    cover. Keeping it simple also keeps the baseline honest.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    """BM25 index over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk to index.")
        self.chunks = chunks
        corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Top-K chunks for `query`, highest score first.

        Uses `argpartition` to find the top K without fully sorting all scores —
        over ~100k chunks and hundreds of queries that difference is the
        difference between minutes and an afternoon.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(query_tokens))
        k = min(top_k, len(scores))

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            RetrievedChunk(chunk=self.chunks[int(i)], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top_idx, start=1)
        ]

    def __len__(self) -> int:
        return len(self.chunks)
