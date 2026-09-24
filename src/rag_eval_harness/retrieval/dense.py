"""Dense retrieval: embedding model plus exact cosine search.

"Dense" is the name of the technique, not a model. Two pieces:

  1. An embedding model turns text into 384 numbers representing meaning.
     Downloaded, already trained, never fine-tuned here.
  2. Cosine similarity finds the closest vectors. Plain arithmetic, no AI.

The search is **exact** — every query is compared against every chunk. No HNSW,
no approximate index. An approximate index has its own recall below 1.0, so
measuring chunking or retrieval through one would mix the method's error with the
index's error inseparably. At 64k chunks x 384 dimensions the full matrix is under
100 MB and an exhaustive search takes milliseconds, so there is nothing to gain by
approximating at measurement time.

Vectors are L2-normalised at encode time, which makes cosine similarity reduce to
a dot product: one matrix multiply for the whole corpus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "retrieval.dense needs `sentence-transformers`. "
        "Install with: uv add sentence-transformers"
    ) from exc

from rag_eval_harness.chunking.fixed import Chunk
from rag_eval_harness.retrieval.bm25 import RetrievedChunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are trained with an instruction prefix on the query side only. The
# model card specifies this exact string for retrieval; omitting it costs a few
# points of recall, and applying it to the passages as well also hurts.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class EmbeddingStats:
    n_chunks: int
    dim: int
    seconds: float
    from_cache: bool
    device: str


class DenseRetriever:
    """Embedding-based retrieval with exact cosine search."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        device: str | None = None,
        use_query_prefix: bool = True,
    ):
        if not chunks:
            raise ValueError("DenseRetriever needs at least one chunk to index.")

        self.chunks = chunks
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_query_prefix = use_query_prefix
        self.cache_dir = cache_dir
        self.stats: EmbeddingStats | None = None

        self._model = SentenceTransformer(model_name, device=device)
        self.device = str(self._model.device)
        self._embeddings: np.ndarray | None = None

    # --- indexing ---------------------------------------------------------

    def _cache_path(self) -> Path | None:
        """Cache file keyed on the model and the exact chunk set.

        The chunk fingerprint matters: chunking with a different size or strategy
        produces different chunks, and silently reusing embeddings built from a
        different set would score one configuration using another's vectors.
        """
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        digest.update(str(len(self.chunks)).encode())
        for c in self.chunks[:: max(1, len(self.chunks) // 500)]:
            digest.update(f"{c.doc_id}:{c.start}:{c.end}".encode())
        tag = digest.hexdigest()[:16]
        return Path(self.cache_dir) / f"emb_{self.model_name.replace('/', '_')}_{tag}.npy"

    def build(self, show_progress: bool = True) -> EmbeddingStats:
        """Embed every chunk, loading from disk when the same set was embedded before."""
        import time

        path = self._cache_path()
        if path is not None and path.exists():
            started = time.perf_counter()
            self._embeddings = np.load(path)
            self.stats = EmbeddingStats(
                n_chunks=len(self.chunks),
                dim=int(self._embeddings.shape[1]),
                seconds=time.perf_counter() - started,
                from_cache=True,
                device=self.device,
            )
            return self.stats

        started = time.perf_counter()
        vectors = self._model.encode(
            [c.text for c in self.chunks],
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # makes cosine == dot product
        ).astype(np.float32)
        elapsed = time.perf_counter() - started

        self._embeddings = vectors
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)

        self.stats = EmbeddingStats(
            n_chunks=len(self.chunks),
            dim=int(vectors.shape[1]),
            seconds=elapsed,
            from_cache=False,
            device=self.device,
        )
        return self.stats

    # --- retrieval --------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        text = (BGE_QUERY_PREFIX + query) if self.use_query_prefix else query
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Top-K chunks by cosine similarity, highest first."""
        if self._embeddings is None:
            raise RuntimeError("call build() before retrieve()")

        q = self.encode_query(query)
        # Both sides normalised, so the dot product is the cosine.
        scores = self._embeddings @ q

        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            RetrievedChunk(chunk=self.chunks[int(i)], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top_idx, start=1)
        ]

    def __len__(self) -> int:
        return len(self.chunks)
