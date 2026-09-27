"""Structure-aware chunkers: recursive and paragraph.

Both respect the document's own boundaries instead of cutting blindly at a token
count. The question Phase 5 asks is whether that respect is worth anything — a
chunk that ends mid-sentence may still retrieve fine, and a benchmark without a
trivial control can quietly conclude the opposite of the truth.

Every chunker here produces the same `Chunk` type as the fixed-size one, with the
same tiling guarantee: chunk N ends exactly where chunk N+1 begins, so every
character of the document belongs to exactly one chunk. Without that, whitespace
between chunks belongs to nothing, and gold spans that include leading or trailing
whitespace become partially unreachable — capping recall for a reason unrelated to
retrieval quality.
"""

from __future__ import annotations

import re

from rag_eval_harness.chunking.fixed import Chunk, DEFAULT_TOKENIZER

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "chunking.structural needs `transformers`. Install with: uv add transformers"
    ) from exc


# Split points in descending order of preference. The recursive splitter tries the
# first, and only falls to the next when a piece is still over budget.
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
LINE_BREAK = re.compile(r"\n")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
WORD_BREAK = re.compile(r"\s+")


class _TokenCounter:
    """Token counting with the embedder's tokenizer, cached per string.

    Structural chunkers call this repeatedly on overlapping candidate pieces, so
    the cache matters: without it, a 300 KB merger agreement is tokenised dozens
    of times over.
    """

    def __init__(self, tokenizer_name: str = DEFAULT_TOKENIZER):
        self.tokenizer_name = tokenizer_name
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text in self._cache:
            return self._cache[text]
        n = len(self._tok(text, add_special_tokens=False, verbose=False)["input_ids"])
        if len(self._cache) < 50_000:
            self._cache[text] = n
        return n


def _tile(doc_id: str, text: str, cut_points: list[int]) -> list[Chunk]:
    """Turn a sorted list of cut positions into tiling chunks.

    `cut_points` are character offsets where one chunk ends and the next begins.
    The first chunk starts at 0 and the last ends at len(text), so the chunks
    cover the document exactly.

    Whitespace-only pieces are folded into the previous chunk rather than dropped,
    because a gold span touching that whitespace would otherwise be unreachable.
    """
    bounds = [0] + sorted(set(cut_points)) + [len(text)]
    chunks: list[Chunk] = []

    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(doc_id=doc_id, start=start, end=end, text=piece))
        elif chunks:
            prev = chunks[-1]
            chunks[-1] = Chunk(
                doc_id=prev.doc_id,
                start=prev.start,
                end=end,
                text=text[prev.start : end],
            )
    return chunks


class RecursiveChunker:
    """Split on the largest natural boundary that fits the budget.

    The common default in RAG tooling. Tries paragraph breaks first, then single
    newlines, then sentence ends, then whitespace — descending only when a piece
    is still over the token budget.

    The intent is that a chunk ends at a paragraph or sentence boundary rather
    than mid-word, so the retrieved text reads as a coherent unit. Whether that
    helps retrieval is the open question; on contracts, where a clause may run
    across several paragraphs, it may not.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        counter: _TokenCounter | None = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.tokenizer_name = tokenizer_name
        self._counter = counter or _TokenCounter(tokenizer_name)

    @property
    def name(self) -> str:
        return "recursive"

    def _split_positions(self, text: str, offset: int, depth: int = 0) -> list[int]:
        """Cut positions for `text`, which starts at `offset` in the document."""
        if self._counter.count(text) <= self.chunk_size_tokens:
            return []

        separators = [PARAGRAPH_BREAK, LINE_BREAK, SENTENCE_BREAK, WORD_BREAK]
        if depth >= len(separators):
            # No separator left. Cut at the midpoint so progress is still made;
            # this only happens on pathological text with no whitespace at all.
            mid = len(text) // 2
            return [offset + mid] if mid > 0 else []

        pattern = separators[depth]
        pieces: list[tuple[int, int]] = []
        last = 0
        for match in pattern.finditer(text):
            if match.end() > last:
                pieces.append((last, match.end()))
                last = match.end()
        if last < len(text):
            pieces.append((last, len(text)))

        if len(pieces) <= 1:
            return self._split_positions(text, offset, depth + 1)

        cuts: list[int] = []
        group_start = 0
        group_end = 0

        for piece_start, piece_end in pieces:
            candidate = text[group_start:piece_end]
            if group_end > group_start and self._counter.count(candidate) > self.chunk_size_tokens:
                cuts.append(offset + group_end)
                # The closed group may itself still be over budget if one piece
                # alone exceeds it; recurse into it at the next separator level.
                closed = text[group_start:group_end]
                if self._counter.count(closed) > self.chunk_size_tokens:
                    cuts.extend(
                        self._split_positions(closed, offset + group_start, depth + 1)
                    )
                group_start = group_end
            group_end = piece_end

        tail = text[group_start:]
        if tail and self._counter.count(tail) > self.chunk_size_tokens:
            cuts.extend(self._split_positions(tail, offset + group_start, depth + 1))

        return cuts

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        return _tile(doc_id, text, self._split_positions(text, 0))


class ParagraphChunker:
    """Respect the document's own paragraph breaks. The trivial control.

    Splits on blank lines and nothing else, merging consecutive paragraphs until
    the budget is reached and splitting any single paragraph that exceeds it.

    This is included specifically as a control. On well-structured documents the
    "naive" approach sometimes beats sophisticated ones, and a chunking ablation
    without a trivial baseline can conclude that a clever method won when in fact
    nothing beat doing the obvious thing.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 256,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        counter: _TokenCounter | None = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.tokenizer_name = tokenizer_name
        self._counter = counter or _TokenCounter(tokenizer_name)

    @property
    def name(self) -> str:
        return "paragraph"

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        # Paragraph boundaries, as character offsets.
        boundaries = [m.end() for m in PARAGRAPH_BREAK.finditer(text)]
        if not boundaries:
            # No blank lines anywhere: fall back to single newlines, and if the
            # document has none of those either, one chunk per budget of words.
            boundaries = [m.end() for m in LINE_BREAK.finditer(text)]

        bounds = [0] + boundaries + [len(text)]
        paragraphs = [(s, e) for s, e in zip(bounds, bounds[1:]) if e > s]

        cuts: list[int] = []
        group_start = 0
        group_end = 0

        for p_start, p_end in paragraphs:
            single = text[p_start:p_end]

            if self._counter.count(single) > self.chunk_size_tokens:
                # This paragraph alone is over budget. Close whatever is open,
                # then split the paragraph on sentences.
                if group_end > group_start:
                    cuts.append(group_end)
                cuts.extend(self._split_long(text, p_start, p_end))
                group_start = p_end
                group_end = p_end
                continue

            candidate = text[group_start:p_end]
            if group_end > group_start and self._counter.count(candidate) > self.chunk_size_tokens:
                cuts.append(group_end)
                group_start = group_end
            group_end = p_end

        return _tile(doc_id, text, cuts)

    def _split_long(self, text: str, start: int, end: int) -> list[int]:
        """Split one oversized paragraph on sentence ends."""
        body = text[start:end]
        cuts: list[int] = []
        group_start = 0
        group_end = 0
        last = 0
        pieces: list[tuple[int, int]] = []

        for match in SENTENCE_BREAK.finditer(body):
            pieces.append((last, match.end()))
            last = match.end()
        if last < len(body):
            pieces.append((last, len(body)))

        if len(pieces) <= 1:
            return cuts

        for p_start, p_end in pieces:
            candidate = body[group_start:p_end]
            if group_end > group_start and self._counter.count(candidate) > self.chunk_size_tokens:
                cuts.append(start + group_end)
                group_start = group_end
            group_end = p_end

        return cuts
