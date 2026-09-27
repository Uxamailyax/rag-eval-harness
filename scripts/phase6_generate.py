"""Phase 6, step 1: generate answers over the golden set.

The retrieval half of the project is complete. This adds the generation half and
makes it a working RAG system: question -> retrieve top 10 chunks -> pack them into
a prompt within the token budget -> the LLM writes an answer from that text alone.

**Context packing uses DROP, and that was measured rather than assumed.**
Groq's free tier caps a request at 8,000 tokens per minute, and ten 512-token chunks
of legal text can exceed it. Two policies were available:

    DROP       add chunks in rank order until the budget is reached, then stop
    TRUNCATE   keep all ten, cut each one proportionally to fit

Convention favours DROP, but the harness already held everything needed to decide it
on this corpus. `scripts/context_packing_experiment.py` simulated both over the 45
answerable golden cases at a 5,100-token budget, where 64% of questions exceed the
limit. The result:

    policy      recall@10   precision@10   mean chars
    no limit       0.3816         0.0072        23,168
    DROP           0.3816         0.0078        23,168
    TRUNCATE       0.3802         0.0072        24,732

**DROP costs nothing.** Identical recall to having no limit at all, while improving
precision 8% and sending 6% fewer characters. The gold was never in chunk 10 — MRR
of 0.22 means that when gold is found it sits near the top, so the lowest-ranked
chunk was never carrying the answer.

TRUNCATE damaged all ten chunks including rank 1, delivered slightly less gold, and
sent *more* text doing it.

**Why the token budget is conservative.** Token counting here uses bge-small's
tokenizer; Groq uses a different one, and on legal vocabulary the two disagree by
20-30%. A budget computed locally can therefore still be rejected by the API. So the
budget is set low, and a 413 is caught and retried with one fewer chunk — the API
gets the final say rather than a tokenizer that does not match it.

**Built to be run repeatedly.** Every answer is appended to disk the moment it
arrives, and re-running skips whatever is already done. Stopping halfway loses
nothing. Hitting the daily limit prints what remains rather than a stack trace.

Run with:  uv run python scripts/phase6_generate.py
           uv run python scripts/phase6_generate.py --limit 10
           uv run python scripts/phase6_generate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, asdict

from groq import APIStatusError

from rag_eval_harness.chunking.fixed import Chunk
from rag_eval_harness.config import settings, write_manifest
from rag_eval_harness.llm import LLMClient
from rag_eval_harness.metrics.retrieval import CharRange, evaluate_query
from rag_eval_harness.retrieval.dense import DenseRetriever

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("needs `transformers`. Install with: uv add transformers") from exc

STRATEGY = "fixed"
SIZE = 512
TOP_K = 10
KS = (5, 10)
TOKENIZER = "BAAI/bge-small-en-v1.5"

EVALS_DIR = settings.data_dir.parent / "evals"
GOLDEN_PATH = EVALS_DIR / "golden_v1.jsonl"
ANSWERS_PATH = EVALS_DIR / "answers_v1.jsonl"
PROMPT_VERSION = "v1"

# Groq free tier for this model.
TPM_LIMIT = 8000
DAILY_TOKEN_BUDGET = 200_000

# Reserved generously because gpt-oss-120b is a reasoning model — Phase 0 measured 60
# output tokens to produce the single word "ping", of which ~56 were internal
# reasoning — and because the local tokenizer undercounts against Groq's.
RESERVED_FOR_OUTPUT = 2500
RESERVED_FOR_SCAFFOLD = 400
CONTEXT_BUDGET = TPM_LIMIT - RESERVED_FOR_OUTPUT - RESERVED_FOR_SCAFFOLD

MAX_TOKENS = 2048

# How many chunks to shed per 413 before giving up on a question.
MAX_SHRINK_ATTEMPTS = 6


SYSTEM_PROMPT = """You are a legal document analyst. Answer questions about contracts \
using only the excerpts provided.

Rules:
- Base your answer solely on the provided excerpts. Do not use outside knowledge.
- If the excerpts do not contain the answer, reply exactly: I don't know.
- Quote the relevant contract language where it supports your answer.
- Be concise. Two or three sentences is usually enough."""

USER_TEMPLATE = """Excerpts from the document:

{context}

---

Question: {question}

Answer using only the excerpts above."""


@dataclass
class AnswerRecord:
    id: str
    question: str
    answerable: bool
    subset: str
    answer: str
    cached: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float | None
    retrieved: list[dict]
    chunks_retrieved: int
    chunks_sent: int
    chunks_dropped: int
    shrink_retries: int
    retrieval_recall_10: float
    retrieval_precision_10: float
    context_chars: int
    context_tokens_est: int
    prompt_version: str
    model: str


class ContextPacker:
    """DROP packing: add chunks in rank order until the budget is reached."""

    def __init__(self, tokenizer_name: str = TOKENIZER):
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._cache:
            self._cache[text] = len(
                self._tok(text, add_special_tokens=False, verbose=False)["input_ids"]
            )
        return self._cache[text]

    def pack(self, hits: list, budget: int, max_chunks: int | None = None) -> list:
        """Chunks that fit, in rank order.

        `max_chunks` caps the count regardless of budget — used by the 413 retry to
        shed a chunk at a time when the API disagrees with the local token count.
        """
        kept = []
        used = 0
        limit = max_chunks if max_chunks is not None else len(hits)

        for hit in hits[:limit]:
            n = self.count(hit.chunk.text)
            if kept and used + n > budget:
                break
            kept.append(hit)
            used += n
        return kept


def build_context(hits: list) -> tuple[str, int]:
    """Format chunks for the prompt, numbered and attributed so an answer can be
    traced back to a specific excerpt."""
    parts = []
    for hit in hits:
        name = hit.chunk.doc_id.split("/")[-1]
        parts.append(f"[Excerpt {hit.rank} — {name}]\n{hit.chunk.text.strip()}")
    context = "\n\n".join(parts)
    return context, len(context)


def load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"golden set missing: {GOLDEN_PATH}\nRun scripts/build_golden_set.py first."
        )
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_chunks() -> list[Chunk]:
    path = settings.cache_path.parent / f"chunks_{STRATEGY}_{SIZE}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"chunk cache missing: {path.name}\nRun scripts/phase5_sweep.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_done() -> dict[str, dict]:
    if not ANSWERS_PATH.exists():
        return {}
    done: dict[str, dict] = {}
    with open(ANSWERS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def generate_with_shrink(
    client: LLMClient, packer: ContextPacker, hits: list, question: str
) -> tuple[object, list, int]:
    """Call the model, shedding a chunk each time the API rejects the request.

    A 413 means the request exceeds what a single minute can carry, so it will never
    succeed no matter how long we wait — unlike a 429, which the client already
    retries with backoff. The only fix is a smaller request.

    This exists because the local tokenizer disagrees with Groq's by 20-30% on legal
    vocabulary, so a prompt that fits the computed budget can still be refused. The
    API gets the final say.
    """
    kept = packer.pack(hits, CONTEXT_BUDGET)

    for attempt in range(MAX_SHRINK_ATTEMPTS):
        context, _ = build_context(kept)
        prompt = USER_TEMPLATE.format(context=context, question=question)
        try:
            response = client.complete(
                prompt,
                model=settings.generator_model,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            return response, kept, attempt
        except APIStatusError as exc:
            if exc.status_code != 413 or len(kept) <= 1:
                raise
            kept = kept[:-1]
            print(f"      413 — retrying with {len(kept)} chunks")

    raise RuntimeError(
        f"still too large after shedding {MAX_SHRINK_ATTEMPTS} chunks"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--golden", default="golden_v1.jsonl",
                        help="which golden set to run")
    args = parser.parse_args()

    global GOLDEN_PATH, ANSWERS_PATH
    GOLDEN_PATH = EVALS_DIR / args.golden
    ANSWERS_PATH = EVALS_DIR / args.golden.replace("golden_", "answers_")

    golden = load_golden()
    done = load_done()
    remaining = [g for g in golden if g["id"] not in done]

    print(f"golden set   : {len(golden)} examples "
          f"({sum(1 for g in golden if not g['answerable'])} unanswerable)")
    print(f"already done : {len(done)}")
    print(f"remaining    : {len(remaining)}")

    if not remaining:
        print(f"\nAll answers generated.\n  {ANSWERS_PATH}")
        return 0

    chunks = load_chunks()
    dense = DenseRetriever(chunks, cache_dir=settings.cache_path.parent)
    stats = dense.build(show_progress=False)
    if not stats.from_cache:
        print("warning: embeddings rebuilt rather than loaded from cache")

    packer = ContextPacker()

    print(f"chunks       : {len(chunks)} ({STRATEGY} @ {SIZE})")
    print(f"model        : {settings.generator_model}")
    print(f"packing      : DROP, {CONTEXT_BUDGET:,}-token context budget")
    print(f"               (TPM {TPM_LIMIT:,} minus {RESERVED_FOR_OUTPUT:,} output "
          f"and {RESERVED_FOR_SCAFFOLD} scaffold)\n")

    # --- estimate ------------------------------------------------------------
    sample = remaining[: min(5, len(remaining))]
    est_tokens = 0
    est_dropped = 0
    for g in sample:
        hits = dense.retrieve(g["question"], top_k=TOP_K)
        kept = packer.pack(hits, CONTEXT_BUDGET)
        est_dropped += len(hits) - len(kept)
        est_tokens += sum(packer.count(h.chunk.text) for h in kept)

    per_q = int(est_tokens / len(sample)) + 600
    total = per_q * len(remaining)

    print("--- estimate ---")
    print(f"  ~{per_q:,} tokens per question")
    print(f"  ~{total:,} tokens for the {len(remaining)} remaining")
    print(f"  ~{est_dropped / len(sample):.1f} chunks dropped per question by packing")
    print(f"  daily free-tier budget: ~{DAILY_TOKEN_BUDGET:,}")
    if total > DAILY_TOKEN_BUDGET:
        print(f"  -> roughly {total / DAILY_TOKEN_BUDGET:.1f} days. Run in batches;")
        print(f"     completed questions are cached and never repeated.")
    print()

    if args.dry_run:
        print("dry run, nothing called.")
        return 0

    target = remaining[: args.limit] if args.limit else remaining
    print(f"--- generating {len(target)} answers ---\n")

    client = LLMClient()
    records: list[AnswerRecord] = []
    session_tokens = 0
    stopped = False
    stop_reason = ""

    for n, g in enumerate(target, start=1):
        hits = dense.retrieve(g["question"], top_k=TOP_K)

        try:
            response, kept, retries = generate_with_shrink(
                client, packer, hits, g["question"]
            )
        except (RuntimeError, APIStatusError) as exc:
            stopped = True
            stop_reason = str(exc)
            print(f"\n  STOPPED at {n - 1}/{len(target)}")
            break

        if not response.cached:
            session_tokens += response.total_tokens or 0

        context, context_chars = build_context(kept)
        context_tokens = sum(packer.count(h.chunk.text) for h in kept)

        # Scored over what was actually sent, not over the full top-10. If packing
        # dropped a chunk carrying gold, the recall recorded here reflects that.
        if g["answerable"]:
            gold = [CharRange(s["doc_id"], s["start"], s["end"]) for s in g["gold_spans"]]
            sent = [CharRange(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in kept]
            scores = evaluate_query(sent, gold, ks=KS)
            r10, p10 = scores["recall@10"], scores["precision@10"]
        else:
            r10 = p10 = 0.0

        record = AnswerRecord(
            id=g["id"],
            question=g["question"],
            answerable=g["answerable"],
            subset=g["subset"],
            answer=response.text.strip(),
            cached=response.cached,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            retrieved=[
                {
                    "rank": h.rank,
                    "doc_id": h.chunk.doc_id,
                    "start": h.chunk.start,
                    "end": h.chunk.end,
                    "score": round(h.score, 4),
                    "sent": h in kept,
                }
                for h in hits
            ],
            chunks_retrieved=len(hits),
            chunks_sent=len(kept),
            chunks_dropped=len(hits) - len(kept),
            shrink_retries=retries,
            retrieval_recall_10=round(r10, 4),
            retrieval_precision_10=round(p10, 4),
            context_chars=context_chars,
            context_tokens_est=context_tokens,
            prompt_version=PROMPT_VERSION,
            model=settings.generator_model,
        )
        records.append(record)

        with open(ANSWERS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        tag = "cache" if response.cached else f"{response.total_tokens or 0} tok"
        mark = "?" if not g["answerable"] else " "
        sent_tag = f"{len(kept)}/{len(hits)}"
        preview = " ".join(record.answer.split())[:58]
        print(f"  [{n}/{len(target)}]{mark} {g['id']:<18} {sent_tag:>6} {tag:>10}  "
              f"{preview}")

        if n % 10 == 0 and not stopped:
            print(f"      ... {session_tokens:,} tokens this session\n")

    # --- summary -------------------------------------------------------------
    total_done = len(done) + len(records)
    left = len(golden) - total_done

    print(f"\n{'=' * 82}")
    if stopped:
        print("RUN STOPPED")
        print("=" * 82)
        print(f"\n  reason: {stop_reason[:200]}")
        print(f"\n  generated this session : {len(records)}")
        print(f"  total complete         : {total_done}/{len(golden)}")
        print(f"  remaining              : {left}")
        print("\n  Everything generated is cached. Resume with the same command —")
        print("  completed questions are skipped, not repeated.")
    else:
        print(f"BATCH COMPLETE — {len(records)} answers generated")
        print("=" * 82)
        print(f"\n  total complete : {total_done}/{len(golden)}")
        print(f"  remaining      : {left}")
        if left:
            print("\n  Run again to continue.")

    new = [r for r in records if not r.cached]
    if new:
        pt = sum(r.prompt_tokens or 0 for r in new)
        ct = sum(r.completion_tokens or 0 for r in new)
        print(f"\n  tokens this session : {pt:,} in / {ct:,} out")
        print(f"  mean per question   : {pt // len(new):,} in / {ct // len(new):,} out")
        print(f"  daily budget used   : {(pt + ct) / DAILY_TOKEN_BUDGET:.0%}")

    if records:
        dropped = sum(r.chunks_dropped for r in records)
        retried = sum(1 for r in records if r.shrink_retries > 0)
        print(f"\n  packing: {dropped} chunks dropped across {len(records)} questions "
              f"({dropped / len(records):.1f} each)")
        if retried:
            print(f"           {retried} questions needed a 413 retry — the local")
            print(f"           tokenizer undercounted against Groq's")

    # --- hallucination signal so far -----------------------------------------
    everything = list(done.values()) + [asdict(r) for r in records]
    unanswerable = [r for r in everything if not r["answerable"]]
    if unanswerable:
        refused = sum(1 for r in unanswerable if "don't know" in r["answer"].lower())
        print(f"\n  unanswerable cases done : {len(unanswerable)}/5")
        print(f"  said \"I don't know\"     : {refused}")
        if refused < len(unanswerable):
            print(f"  -> {len(unanswerable) - refused} answered anyway. That is the")
            print("     hallucination Phase 6 exists to measure.")

    answered = [r for r in everything if r["answerable"]]
    if answered:
        zero_recall = [r for r in answered if r["retrieval_recall_10"] == 0]
        refused_zero = sum(
            1 for r in zero_recall if "don't know" in r["answer"].lower()
        )
        if zero_recall:
            print(f"\n  answerable questions where retrieval found no gold: "
                  f"{len(zero_recall)}")
            print(f"  of those, the model refused rather than invented: {refused_zero}")
            print("  -> refusing on zero recall is correct behaviour. Inventing an")
            print("     answer there is the failure faithfulness will catch.")

    if not stopped and left == 0:
        write_manifest(
            settings,
            {
                "phase": 6,
                "step": "generate",
                "model": settings.generator_model,
                "prompt_version": PROMPT_VERSION,
                "packing": "drop",
                "context_budget_tokens": CONTEXT_BUDGET,
                "top_k": TOP_K,
                "chunking": f"{STRATEGY}@{SIZE}",
                "golden_set_size": len(golden),
                "answers_path": str(ANSWERS_PATH),
            },
        )

    print(f"\n  answers: {ANSWERS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
