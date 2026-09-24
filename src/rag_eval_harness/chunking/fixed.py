"""Fixed-size chunking by token count.

Two decisions here matter for every number downstream.

**Token counting uses the embedder's own tokenizer.** A "256-token chunk" only
means something in a specific tokenizer. Using `len(text.split())` in Phase 3 and
the embedder's tokenizer in Phase 4 would mean the two phases chunk differently
and their results are not comparable.

**Chunks tile the document exactly.** Each chunk ends where the next begins, so
every character of the source belongs to exactly one chunk. The naive version —
ending a chunk at its last token's final character — leaves the whitespace between
tokens belonging to no chunk at all. Gold spans routinely include leading or
trailing whitespace, so those gaps would cap recall below 1.0 for a reason that has
nothing to do with retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "chunking.fixed needs `transformers` for the embedder's tokenizer. "
        "Install with: uv add transformers"
    ) from exc

DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Chunk:
    """A slice of one document, with offsets into that document's decoded text.

    Field names match `Span` and `CharRange` deliberately, so a chunk converts to
    a `CharRange` with no translation layer to get wrong.
    """

    doc_id: str
    start: int  # inclusive
    end: int  # exclusive
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


class FixedSizeChunker:
    """Split documents into fixed token-count chunks with no overlap."""

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        tokenizer_name: str = DEFAULT_TOKENIZER,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        """Split `text` into chunks of `chunk_size_tokens`, tiling exactly.

        Offsets come from the tokenizer's own `offset_mapping`, so token
        boundaries map back to character positions without re-decoding.
        """
        if not text:
            return []

        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        offsets = encoding["offset_mapping"]
        if not offsets:
            return []

        chunks: list[Chunk] = []
        i = 0
        n = len(offsets)

        while i < n:
            j = min(i + self.chunk_size_tokens, n)

            # Start where the previous chunk ended, so no character is orphaned.
            # The first chunk starts at 0 rather than at the first token, which
            # picks up any leading whitespace in the document.
            start = chunks[-1].end if chunks else 0

            # End at the next chunk's first token, or at the end of the document
            # for the final chunk. This is what makes the tiling exact.
            end = offsets[j][0] if j < n else len(text)

            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(doc_id=doc_id, start=start, end=end, text=chunk_text))
            elif chunks:
                # A whitespace-only run still has to belong somewhere, or a gold
                # span touching it would be unreachable. Extend the previous chunk.
                prev = chunks[-1]
                chunks[-1] = Chunk(
                    doc_id=prev.doc_id,
                    start=prev.start,
                    end=end,
                    text=text[prev.start : end],
                )

            i = j

        return chunks

    def chunk_many(self, documents: dict[str, str]) -> list[Chunk]:
        """Chunk several documents. Returns one flat list."""
        out: list[Chunk] = []
        for doc_id, text in documents.items():
            out.extend(self.chunk(doc_id, text))
        return out
