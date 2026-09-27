# Decisions

Each entry records what was chosen, why, and **what would change it**. The last part is
the one that matters: a decision without a reversal condition is a preference.

---

## Corpus: LegalBench-RAG

Expert-annotated **character-level** retrieval ground truth across 714 documents and
6,858 query→answer pairs. Hand-labelling retrieval spans costs 15–30 minutes per example;
this supplies them for free, which is the difference between measuring retrieval and
guessing at it.

Rejected: Paul Graham essays and "chat with your PDF" (no ground truth, and the most
overused RAG tutorial corpora), Wikipedia (too well-structured to be representative), SEC
10-K (good documents, no labels — costs days to annotate).

**Would change if:** the project needed multi-hop or cross-document questions, which this
benchmark does not contain. ACORD would be the alternative — clause-level rather than
character-level, but with graded 1–5 relevance rather than binary.

---

## Metrics implemented rather than imported

No library scores at character level, and chunk-level scoring makes chunking
configurations incomparable: a 512-token chunk is mechanically more likely to overlap a
gold span than a 128-token chunk, so recall rises with chunk size as a pure artifact.

There is also an interview-shaped reason. "I used RAGAS" and "here is why NDCG's ideal
ranking is the retrieved set's own grades sorted, and what that hides" are different
claims, and only one survives a follow-up question.

61 known-answer tests, expected values computed by hand before the code existed, NDCG
cross-checked against `sklearn.metrics.ndcg_score` on binary relevance — the case where
linear and exponential gain formulations coincide.

**Would change if:** the harness needed metrics beyond retrieval and faithfulness —
context precision, answer relevance, semantic similarity. Re-implementing a metric suite
is not the work; validating one is.

---

## Exact numpy search for measurement, pgvector deferred

HNSW is approximate and has its own recall below 1.0. Measuring chunking through an
approximate index measures chunking **plus** index error, inseparably, and there is no
way to attribute a difference afterwards.

At 32,240 chunks × 384 dimensions the full matrix is under 100 MB and exhaustive search
takes 13 ms. There is no performance argument for approximating at measurement time.

The plan called for quantifying the HNSW-vs-exact gap. That was not done, and the README
records it as unfinished rather than implying it.

**Would change if:** the corpus reached a scale where exact search stopped being
instant — around a million chunks. The measurement path would still want exact search on
a sample, with the approximate index measured against it rather than substituted for it.

---

## No LangChain, no LlamaIndex, no RAGAS

The retrieval loop is roughly 200 lines. Wrapping it in a framework would hide exactly
the mechanics the project exists to demonstrate — how RRF fails, why precision moves
inversely to chunk size, what a cross-encoder actually does with 50 candidates.

RAGAS specifically was dropped for a methodological reason rather than a stylistic one:
Phase 7 calibrates a judge and reports κ. If the faithfulness *number* came from RAGAS
but κ came from a different rubric, **κ would not describe the number being reported.**
The calibration has to apply to the instrument actually used.

**Would change if:** this were production code with a team maintaining it. A framework's
value is other people's familiarity with it, which is worth more than transparency once
more than one person touches the code.

---

## Generator and judge from different model families

`openai/gpt-oss-120b` generates, `qwen/qwen3.8-27b` judges. Models rate their own output
higher than other models' — a model grading itself inflates scores in a way that is
invisible in the numbers.

A secondary benefit made the project affordable: Groq rate-limits per model, so the two
roles draw on separate token budgets.

**Would change if:** a domain-specific legal judge existed. Finding 4 suggests the
reranker's failure is missing domain knowledge rather than the two-stage pattern, and the
same likely applies to judging. Cross-family would still be preferable, but a legal
model from the same family would probably beat a general model from a different one.

---

## Chunk sizes 128 / 256 / 512, not 1024

`bge-small-en-v1.5` has a 512-token maximum sequence length. A 1024-token chunk is
silently truncated at 512 — no error, no warning, and the back half invisible to dense
retrieval while fully visible to BM25. That would have produced a confident and wrong
conclusion that large chunks hurt dense retrieval.

Phase 5d measured the cost rather than assuming it: identical 1024-token chunks scored
**0.4856 on bge-small (reading half) and 0.5924 on BGE-M3 (reading all)** — 22% of recall
lost to silent truncation.

**Would change if:** BGE-M3 were the production embedder. Its 8,192-token window makes
larger chunks a genuine option, and recall was still accelerating at 1024 (+0.169 over
512).

---

## Shipping bge-small @ 512 over BGE-M3 @ 1024

BGE-M3 at 1024 tokens wins decisively on retrieval: **recall +40%, MRR +53%, NDCG +44%**.
It is not shipped.

| | bge-small @ 512 | BGE-M3 @ 1024 |
|---|---|---|
| recall@10 | 0.423 | **0.592** |
| precision@10 | **0.0117** | 0.0075 |
| chars returned per query | **~24,700** | ~49,000 |
| query latency | **13 ms** | 37 ms |
| corpus embedding time | **12 min** | **6 h 8 min** |

The deciding factor is downstream. Retrieved text becomes *input* to a generator billed
per token, so doubling the characters doubles the cost and time-to-first-token of the one
expensive stage — on every query, permanently. Embeddings are built once; generation is
paid forever.

**Would change if:** this were retrieval-only — search, discovery, citation-finding with
a human reading the results. Then BGE-M3 @ 1024 ships and the precision loss is
irrelevant.

---

## Reranker measured, not shipped

`bge-reranker-base` captures 20.5% of available headroom at 128 tokens and **1.8% at
512**, for 16.4 seconds of added latency per query. It also loses 6% of MRR and 6% of
precision at 512.

It is the only intervention that moved the preamble rate — down 10 points at every size —
which is worth stating separately. A mitigation that works and costs too much is a
different finding from one that does not work, and leads to different next steps.

**Would change if:** the workload were offline batch, where 16 s per query stops
mattering. Even then the MRR and precision losses at 512 would argue against it, and the
case would be stronger at 128. Or if a legal-domain cross-encoder were available.

---

## Reranker off during the chunking sweep

With the reranker running, a configuration could win because its chunking suits the
reranker rather than because the chunking is good — "paragraph @ 512 is best" when the
truth is "the reranker likes long chunks." Confounded, and confidently wrong.

Tested separately at three sizes afterwards, because a cross-encoder reads query and
chunk together under a 512-token limit and the amount of context per judgement plausibly
interacts with chunk size. It did: the gain fell an order of magnitude from 128 to 512.

**Would change if:** the reranker were always-on in production. Then the sweep should
measure the system as deployed, and the attribution problem is accepted deliberately
rather than avoided.

---

## DROP context packing over TRUNCATE

Groq's free tier caps requests at 8,000 TPM and 64% of queries exceed the budget at
top-10. Convention favours dropping whole chunks; measuring it produced a stronger claim
than citing convention would have.

| policy | recall@10 | precision@10 | mean chars |
|---|---|---|---|
| no limit | 0.3816 | 0.0072 | 23,168 |
| **DROP** | **0.3816** | **0.0078** | **23,168** |
| TRUNCATE | 0.3802 | 0.0072 | 24,732 |

DROP costs **nothing** — identical recall to having no limit, 8% better precision, 6%
fewer characters. The gold was never in chunk 10, because MRR of 0.22 means it sits near
the top when found at all.

One honest limitation: character recall counts positions, not readability. A chunk cut
mid-sentence scores identically to a whole clause even though the model may be unable to
use it. That asymmetry favours DROP beyond what the numbers show.

**Would change if:** retrieval improved enough that gold routinely appeared at rank 8–10.
Then dropping the tail would start costing recall and the trade would need re-measuring.

---

## Unanswerable cases by cross-domain repointing

The first construction pointed contract questions at *other contracts* where the
benchmark recorded no annotated span of that type. Unsound: CUAD annotates 41 categories,
and a document labelled for 19 was only **checked** for those 19. Absence of a label is
not absence of a clause.

> Annotators claimed these answers are here. They never claimed no other answers are here
> — and that claim is not feasible to make about a 300 KB contract.

Rebuilt by pointing questions at a different document *type*: a contract question at a
mobile app privacy policy, which has no parties, no clauses and no governing law by
construction. Absence becomes structural rather than unobserved, and requires no reading
to verify.

The result flipped from **0/5 refused to 4/5**. The original five were largely correct
answers to questions the benchmark never checked for.

**Would change if:** a corpus offered exhaustive annotation — every clause type checked in
every document. Then same-domain repointing would be sound and the test harder, because
the mismatch would not be obvious from the document type alone.

---

## Refusals count as faithful

"I don't know" asserts nothing unsupported. Decided before labelling and applied
consistently, because a judge disagreeing on one whole class would depress κ for a reason
unrelated to hallucination detection.

The distinction that matters, and that a naive refusal detector gets wrong:

| | | |
|---|---|---|
| "I don't know" | declines to claim | faithful |
| "The excerpts do not contain X" | true statement about the excerpts | faithful |
| "The document does not permit X" | **a claim about the document** | needs support |

All three mention absence; only the third asserts it. A regex matching refusal-like
phrasing swallowed two genuine negative assertions and would have counted them faithful
without the judge seeing them.

**Would change if:** the application penalised refusal — a system that must always answer.
Then "I don't know" becomes a different kind of failure and needs its own metric rather
than being folded into faithfulness.

---

## Deterministic prechecks kept narrow

The plan suggested prechecks could cut judge cost 80–90%: JSON validation, refusal
regex, empty-answer detection. Only one was kept — an exact match on a bare "I don't
know" and nothing else.

A broader refusal regex, written to catch paraphrases, misclassified two confident
negative assertions as refusals. **A cheap check that is wrong is worse than no check**,
because it removes the case from scrutiny entirely rather than merely failing to help.

Even narrow it resolved 56% of answers, because most refusals use the exact phrase the
prompt requested.

**Would change if:** answer volume made judge cost binding. Then a broader precheck
becomes worth building, but it would need its own validation against human labels —
the same treatment the judge got.

---

## CI gates stored results rather than re-running the pipeline

Re-generating 50 answers and re-judging them on every push would exhaust a free-tier
quota within days, and a gate that is expensive to run gets disabled.

Instead the gate reads `evals/faithfulness_v2.jsonl`, which the pipeline regenerates
whenever anything affecting it changes — prompt text is part of the LLM cache key, so
editing a prompt invalidates exactly the calls it affects and nothing else. Free when
nothing changed, correct when something did.

Prompt version and judge model are asserted **separately** from quality, because a score
produced under a different rubric is not comparable to the baseline. That distinguishes
"quality dropped" from "the prompt changed, re-baseline deliberately" rather than
conflating them into a tolerance argument.

**Would change if:** the harness ran against a live system rather than a fixed golden
set. Then CI would need to sample real traffic, and the baseline would need resampling
on a schedule rather than on demand.

---

## 5% regression tolerance

The generator is non-deterministic even at temperature 0, so scores wobble between runs.
A gate that fires on noise gets switched off, which is worse than having no gate.

5% is deliberately permissive for an immature system. The plan's guidance applies: start
loose and tighten as the system stabilises.

**Would change if:** run-to-run variance were measured rather than assumed. Ten repeat
runs would give a standard deviation, and the tolerance could be set at 2σ instead of a
round number.

---

## Contract text stripped from published evaluation files

The LegalBench-RAG repository is MIT, but its documents come from ContractNLI, CUAD, MAUD
and PrivacyQA, each with its own usage policy. Redistribution rights could not be
established, so 149,275 characters of verbatim contract text were removed from the
evaluation files.

Nothing is lost. Every span is stored as `(doc_id, start, end)`, which is how the harness
reads them — Phase 1 verified 10,637 resolve correctly from offsets alone. The regression
gate passes identically on stripped files, which demonstrates the metrics never depended
on the text.

**Would change if:** the corpus licensing were clarified as permitting redistribution, or
the harness were pointed at a corpus that is freely redistributable.

---

## Docker runs the gate, not the pipeline

`docker compose up` runs 71 tests with no corpus, no API key and no network. The heavier
services are behind a profile and must be named explicitly.

A clean clone should verify something in two minutes rather than requiring a 90 MB
download and a Groq account before anything works. The pgvector service the plan
originally specified was dropped — all retrieval is exact numpy, and shipping a database
that nothing queries would have been infrastructure built to satisfy a plan item.

**Would change if:** pgvector were actually used, which would happen at the corpus scale
where exact search stops being instant.

---

## What I would do next

**Two-stage retrieval.** The oracle measured a 65% recall gain and a doubling of
precision from filtering to the correct document first. The real version needs
fuzzy-matching a document description against filenames, which is fiddly and error-prone
— but the oracle establishes it is worth building, and each stage would carry its own
metric so a regression points at the stage that caused it.

**Measure run-to-run variance.** The 5% CI tolerance is a guess. Ten repeat runs would
replace it with a number.

**Quantify the HNSW-vs-exact gap.** Planned, not done. "Approximate search cost N points
of recall for an M× speedup" is a cheap finding once Postgres is running.

**Expand the human label set.** 30 gives a 95% CI of 0.67–1.00. Sixty would roughly halve
the interval width, at the cost of another labelling session.

**Point the harness at a second corpus.** `GroundTruthSource` was built as an interface
precisely so the evaluation code never touches LegalBench-RAG directly. Whether the
findings hold on a different domain is the obvious next question, and the boundary exists
to answer it.
