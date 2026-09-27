"""Faithfulness judge.

Grades whether every claim in an answer is supported by the excerpts it was given.
Used in Phase 7 to measure agreement with human labels, and in Phase 8 to score
answers nobody will ever label by hand.

**Why the judge is a different model from the generator.** Models rate their own
output higher than other models' output — self-preference bias. A model grading
itself inflates scores in a way that is invisible in the numbers. The generator is
`openai/gpt-oss-120b`; the judge is `qwen/qwen3.6-27b`. Different family, different
training data, and a separate Groq rate-limit pool, which is what makes the calibration
run affordable on a free tier.

**Why the rubric names three types of unfaithfulness rather than one.** Hand-labelling
thirty cases produced three distinct failures, and a judge told only about the first
would pass the other two:

    1. UNSUPPORTED   the answer asserts something the excerpts do not contain
    2. CONTRADICTED  the excerpts say the opposite of what the answer claims
    3. UNSUPPORTED   the answer asserts an absence ("there is no X") that the
       ABSENCE     excerpts cannot establish, because they are a partial view

Type 3 is the one that needs saying explicitly. Without it, a judge sees "the answer
says there is no liability cap, and I see no liability cap in the excerpts" and marks
it faithful. That reasoning is wrong: ten chunks of a 300 KB contract cannot establish
what the contract does not contain.

**Why refusals are faithful.** "I don't know" asserts nothing unsupported. This has to
be stated in the rubric because the distinction is narrow and the human labels applied
it consistently:

    "I don't know"                          declines to claim       faithful
    "The excerpts do not contain X"         true about the excerpts faithful
    "The document does not permit X"        a claim about the       needs support
                                            contract

**Position and verbosity controls.** The excerpts are presented in a fixed order and
the answer after them, so every case is scored under identical structure. Verbosity
bias is measured rather than prevented — the correlation between answer length and
verdict is reported, because a judge that simply prefers longer answers would show it
there.

**Faithfulness is not correctness.** An answer can be true about the world and
unfaithful, if the support is not in the context provided. The rubric says so, because
a judge that drifts into fact-checking is measuring something else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_eval_harness.config import settings
from rag_eval_harness.llm import LLMClient

JUDGE_PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are evaluating whether an answer is FAITHFUL to the excerpts it \
was given.

Faithful means: every factual claim in the answer is supported by the excerpts shown. \
It does NOT mean the answer is correct about the world. An answer can be factually \
true and still unfaithful, if the support is not in the excerpts provided.

Mark the answer UNFAITHFUL if any of these apply:

1. UNSUPPORTED — the answer asserts something the excerpts do not contain. This \
includes answers that describe a document none of the excerpts are from.

2. CONTRADICTED — the excerpts say the opposite of what the answer claims, or contain \
a qualification the answer ignores. Quoting accurately but omitting a carve-out that \
changes the meaning counts here.

3. UNSUPPORTED ABSENCE — the answer asserts that something is NOT in the document \
("there is no X", "the agreement does not permit X"), when the excerpts cannot \
establish that. The excerpts are a small, partial selection from a long document. Not \
finding something in them is not evidence it is absent from the document. A claim of \
absence needs the same grounding as a claim of presence — an explicit statement that \
none exists, or an exhaustive list.

Mark the answer FAITHFUL if:

- Every claim traces to the excerpts, OR
- The answer declines to answer ("I don't know", "the excerpts do not contain enough \
information"). A refusal asserts nothing unsupported, so it is faithful.

Note the distinction carefully:
  "I don't know"                        -> faithful, declines to claim
  "The excerpts do not contain X"       -> faithful, a true statement about the excerpts
  "The document does not permit X"      -> a claim about the document, needs support

Respond with JSON only, no other text:
{"verdict": "faithful" or "unfaithful", "type": "supported" or "unsupported" or \
"contradicted" or "unsupported_absence" or "refusal", "reason": "one sentence"}"""

USER_TEMPLATE = """EXCERPTS PROVIDED TO THE ANSWERER:

{context}

---

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

---

Is every claim in this answer supported by the excerpts above? Respond with JSON only."""


@dataclass(frozen=True)
class Verdict:
    faithful: bool
    failure_type: str
    reason: str
    raw: str
    cached: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float | None
    precheck: str | None = None


# Deterministic pre-checks. Reported in the plan to cut judge cost 80-90% without
# losing detection rate — but only applied to cases where the verdict is certain
# without reading the excerpts, which in practice is the literal refusal phrase and
# nothing else. Anything more ambitious risks the classifier error from Phase 6,
# where a pattern match on "does not contain" swallowed genuine assertions.
CERTAIN_REFUSAL = re.compile(
    r"^\s*(i don'?t know\.?|i do not know\.?)\s*$", re.IGNORECASE
)


def precheck(answer: str) -> Verdict | None:
    """Return a verdict without calling the judge, when one is certain.

    Deliberately narrow. A bare "I don't know" asserts nothing and cannot be
    unfaithful under any reading. Everything else goes to the judge, because Phase 6
    established that pattern matching on refusal-like phrasing misclassifies
    confident negative claims as refusals.
    """
    if CERTAIN_REFUSAL.match(answer.strip()):
        return Verdict(
            faithful=True,
            failure_type="refusal",
            reason="bare refusal, asserts nothing",
            raw="",
            cached=True,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            precheck="bare_refusal",
        )
    return None


def parse_verdict(text: str) -> tuple[bool, str, str]:
    """Extract the verdict from the model's JSON, tolerating fences and preamble."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).lower()
            return (
                verdict.startswith("faith"),
                str(data.get("type", "unknown")),
                str(data.get("reason", "")),
            )
        except json.JSONDecodeError:
            pass

    # Fall back to reading the words. A judge that cannot produce JSON is still
    # giving an opinion, and discarding it would silently drop cases.
    lowered = cleaned.lower()
    if "unfaithful" in lowered:
        return False, "unknown", "parsed from text, not JSON"
    if "faithful" in lowered:
        return True, "unknown", "parsed from text, not JSON"
    return False, "unparseable", f"could not parse: {cleaned[:120]}"


class FaithfulnessJudge:
    """Grades answers for faithfulness against the excerpts they were given."""

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str | None = None,
        use_prechecks: bool = True,
    ):
        self.client = client or LLMClient()
        self.model = model or settings.judge_model
        self.use_prechecks = use_prechecks
        self.precheck_hits = 0
        self.judge_calls = 0

    def build_context(self, retrieved: list[dict], source) -> str:
        """Reconstruct the excerpt text the answerer saw, from character offsets."""
        parts = []
        for hit in retrieved:
            name = hit["doc_id"].split("/")[-1]
            try:
                text = source.document_text(hit["doc_id"])[hit["start"] : hit["end"]]
            except FileNotFoundError:
                continue
            parts.append(f"[Excerpt {hit['rank']} — {name}]\n{text.strip()}")
        return "\n\n".join(parts)

    def judge(
        self,
        question: str,
        answer: str,
        context: str,
        max_tokens: int = 1024,
    ) -> Verdict:
        if self.use_prechecks:
            early = precheck(answer)
            if early is not None:
                self.precheck_hits += 1
                return early

        prompt = USER_TEMPLATE.format(
            context=context, question=question, answer=answer
        )
        response = self.client.complete(
            prompt,
            model=self.model,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if not response.cached:
            self.judge_calls += 1

        faithful, failure_type, reason = parse_verdict(response.text)

        return Verdict(
            faithful=faithful,
            failure_type=failure_type,
            reason=reason,
            raw=response.text,
            cached=response.cached,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )

    def stats(self) -> dict:
        total = self.precheck_hits + self.judge_calls
        return {
            "precheck_hits": self.precheck_hits,
            "judge_calls": self.judge_calls,
            "precheck_rate": round(self.precheck_hits / total, 4) if total else 0.0,
        }
