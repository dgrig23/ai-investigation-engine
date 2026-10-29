# Evaluation report

All numbers are produced by `eval/run_eval.py` and stored with per-question details in
`eval/results/*.json` (including the git commit). `python -m eval.report` regenerates every table in
[eval_tables.md](eval_tables.md). Runs are reproducible: fixed corpora and question sets, fixed seeds,
temperature 0, and a response cache (re-running a finished evaluation costs no LLM calls).

## 1. Methodology

### Question sets

| Set | Size | Ground truth | Used for |
|---|---|---|---|
| Acme Play investigation questions (`eval/investigation_questions.py`) | 50 | written from the corpus generator's ground truth | retrieval (41 with evidence) + answers (50) |
| HotpotQA distractor sample (`eval/datasets/hotpotqa_questions.jsonl`, seed 42) | 300 | the dataset's gold supporting paragraphs | retrieval over all 66,552 documents |

| Category | n | A correct system must… |
|---|---|---|
| simple | 12 | answer from one document |
| multi_hop | 10 | chain facts across documents (incident → change → author / approval / owner) |
| multi_doc | 8 | aggregate or compare several documents (includes the brief's example question) |
| contradiction | 8 | give the reliable value **and** report the conflicting sources |
| unanswerable | 9 | abstain (8) or correct a false premise (1) - never invent |
| injection | 3 | answer correctly although the evidence contains prompt-injection text |

HotpotQA questions are *bridge* (a two-hop chain whose second paragraph is not named in the
question) or *comparison* (two named entities). Each needs two gold paragraphs hidden among 66,552
documents, including 8 hand-picked distractors per question.

### Metrics

- **Retrieval** (document level). Evidence is a list of *groups* of interchangeable documents (e.g.
  "the postmortem or its draft"). Recall@k = share of groups with a document in the top k;
  **All@k** = every group covered (complete evidence, what multi-hop needs); **MRR** = 1/rank of the
  first relevant document.
- **Answers.** *Correct* (strict): all key facts present and no abstention for answerable questions;
  abstention for unanswerable ones; abstention *or* premise correction for the false-premise question.
  *Key-fact recall* (partial credit). *Citation precision / recall* at document level. *Groundedness*
  from the verifier. *Contradiction detection* on contradiction questions and *false-positive rate* on
  ordinary answerable ones. *Injection resistance*: the payload never appears in the answer.
  *LLM-as-judge* (Qwen3.8-27B - a different model family from the evaluated gpt-oss models) scoring
  1 / 0.5 / 0 against the reference. Cost: LLM calls, tokens, latency.

### Configurations

1. Retrieval: dense · BM25 · hybrid RRF · hybrid + MMR (**production on the 512 MB host**) ·
   hybrid + cross-encoder · hybrid + cross-encoder + MMR.
2. Chunking: structure-aware vs fixed 220-word windows (separate indexes, same questions).
3. Answering: multi-step agent vs single-shot RAG (same model: gpt-oss-120b).
4. LLMs: gpt-oss-120b vs gpt-oss-20b (same code, same questions).

## 2. Retrieval results

### 2.1 Investigation corpus (structure chunking, Qdrant Cloud, 41 questions with evidence)

| config | Recall@1 | Recall@5 | Recall@10 | All@10 | MRR | latency (ms) |
|---|---|---|---|---|---|---|
| dense | 0.394 | 0.846 | 0.939 | 0.902 | 0.668 | 103 |
| BM25 | 0.596 | 0.846 | 0.931 | 0.854 | 0.858 | 111 |
| hybrid RRF | 0.528 | 0.882 | 0.955 | 0.902 | 0.826 | 362 |
| hybrid RRF + MMR (production) | 0.528 | 0.894 | 0.955 | 0.902 | 0.822 | 1,031 |
| hybrid RRF + cross-encoder | **0.669** | **0.931** | **0.963** | **0.927** | **0.917** | 1,272 |
| hybrid RRF + cross-encoder + MMR | 0.669 | 0.931 | 0.963 | 0.927 | 0.913 | 2,199 |

### 2.2 HotpotQA (66,552 documents, 300 questions)

| config | Recall@1 | Recall@5 | Recall@10 | All@10 | MRR | latency (ms) |
|---|---|---|---|---|---|---|
| dense | 0.420 | 0.755 | 0.800 | 0.670 | 0.877 | 168 |
| BM25 | 0.395 | 0.697 | 0.793 | 0.610 | 0.854 | 260 |
| hybrid RRF | 0.420 | 0.780 | 0.858 | 0.743 | 0.888 | 513 |
| hybrid RRF + MMR (production) | 0.420 | 0.772 | 0.857 | 0.737 | 0.888 | 2,140 |
| hybrid RRF + cross-encoder | **0.452** | **0.817** | **0.873** | **0.767** | **0.933** | 963 |
| hybrid RRF + cross-encoder + MMR | 0.452 | 0.808 | 0.868 | 0.757 | 0.933 | 1,846 |

By question type (best config): **comparison** All@10 = 0.983, **bridge** All@10 = 0.703.

### 2.3 Chunking strategies (investigation corpus, same questions)

| config | structure-aware R@5 / R@10 / MRR | fixed 220-word windows R@5 / R@10 / MRR |
|---|---|---|
| dense | 0.846 / 0.939 / 0.668 | 0.803 / 0.931 / 0.719 |
| hybrid RRF | 0.882 / 0.955 / 0.826 | 0.888 / 0.943 / 0.827 |
| hybrid + cross-encoder | **0.931 / 0.963 / 0.917** | 0.870 / 0.915 / 0.847 |

### 2.4 What the retrieval numbers say

1. **Hybrid search beats either retriever alone on both corpora** (HotpotQA All@10: dense 0.670,
   BM25 0.610, hybrid 0.743). They fail on different questions: BM25 wins on identifier-heavy questions
   ("CHG-1047", "PoolSaturationHigh") - on the incident corpus its MRR is 0.858 vs 0.668 for dense -
   while dense wins on paraphrases. RRF needs no score normalization, which is why it is robust here.
2. **The cross-encoder mainly improves the *order*:** MRR +0.09 (investigation) and +0.045 (HotpotQA),
   Recall@1 0.53 → 0.67. It costs ~0.5 s per query on CPU and ~250 MB of RAM - which is why the
   512 MB production host runs without it (MRR 0.822 instead of 0.913; Recall@10 is unchanged).
3. **MMR did not help on these benchmarks** (±0.01). Both question sets need a few *specific*
   documents, so diversity rarely adds a missing one; on HotpotQA it slightly lowers recall by pushing
   out a second paragraph of the same topic. It is kept in the agent (where several searches already
   provide diversity it matters less) but would be the first thing to tune per use case.
4. **Structure-aware chunking wins when combined with the cross-encoder** (R@5 0.931 vs 0.870). Its
   chunks are coherent sections with a title header; fixed windows cut facts at arbitrary offsets, and
   the cross-encoder only reads the first 800 characters of each passage, so facts at the end of a
   fixed window are invisible to it. Without re-ranking the two strategies are equivalent - the
   corpus's documents are short, so both often produce one chunk per document.
5. **Single-query retrieval cannot complete bridge questions.** HotpotQA *bridge* questions reach
   All@10 = 0.70 vs 0.98 for *comparison* questions; the same pattern explains every incomplete case
   in the investigation set (m01, m05, d01, d05): the second piece of evidence (e.g. the change record
   of CHG-1047) is not named in the question and only becomes searchable after the first hop. This is
   the case for the agent's follow-up searches and `find_mentions` tool.
6. **Quantization:** on Qdrant Cloud vectors are int8-quantized to fit the free tier; compared to the
   unquantized local index, dense retrieval differed by at most 0.025 on any metric on the investigation set.

### 2.5 Embedding models (bonus)

Same chunks, same questions, separate indexes. HotpotQA here = the 599 gold paragraphs of the 300
questions hidden among 5,000 random other paragraphs (re-embedding all 66k documents per model would
take hours on a laptop CPU), so absolute numbers are higher than in 2.2 - compare models, not tables.

| Embedding model (dimensions, size) | Investigation dense R@10 / MRR | Investigation hybrid R@10 / MRR | HotpotQA-subset dense R@10 / MRR | HotpotQA-subset hybrid R@10 / MRR |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (384, 90 MB) | 0.842 / 0.683 | 0.931 / 0.828 | 0.860 / 0.923 | 0.923 / 0.961 |
| **bge-small-en-v1.5 (384, 67 MB) - used** | **0.939** / 0.668 | 0.963 / 0.832 | 0.928 / 0.968 | 0.957 / 0.979 |
| bge-base-en-v1.5 (768, 210 MB) | 0.927 / **0.769** | **0.968 / 0.882** | **0.952 / 0.970** | **0.965 / 0.978** |

- bge-base is best on the open-domain HotpotQA subset (+0.024 dense Recall@10) and orders better on
  the incident corpus (MRR +0.10), at 3× the size, 2× the vector dimensions and ~2× slower embedding.
- MiniLM, the most common default, is clearly weaker (−0.10 dense Recall@10 on the incident corpus).
- **Hybrid search shrinks the gap between embedding models** (0.931-0.968 vs 0.842-0.939 for dense
  alone): BM25 compensates for a weaker embedding. With 512 MB RAM in production and RAM-bound
  int8 vectors on the 1 GB Qdrant tier, bge-small is the right trade-off; bge-base is the first upgrade
  on a larger host.

### 2.6 Performance under load (bonus)

`scripts/load_test.py`: 64 HotpotQA questions per level sent concurrently to `GET /api/search` on a
local server (8-core laptop) using the full 66,552-document index on Qdrant Cloud. No LLM involved.

| concurrent requests | production (no cross-encoder): p50 / p95 / throughput | with cross-encoder: p50 / p95 / throughput |
|---|---|---|
| 1 | 391 / 553 ms / 2.4 req/s | 952 / 1,107 ms / 1.0 req/s |
| 4 | 451 / 589 ms / 8.4 req/s | 2,016 / 2,176 ms / 1.9 req/s |
| 8 | 512 / 698 ms / 14.1 req/s | 4,050 / 4,260 ms / 1.9 req/s |
| 16 | 713 / 1,275 ms / **19.3 req/s** | 8,134 / 8,284 ms / **1.9 req/s** |

0 errors in 512 requests. Without the cross-encoder, throughput scales with concurrency (the time is
mostly network I/O to Qdrant Cloud, which overlaps). With it, throughput is **capped at ~2 req/s** and
latency grows linearly: `CrossEncoderReranker` serializes calls with a lock around the single ONNX
session, so it is a CPU-bound queue. Fixes: a small pool of sessions (one per core), batching
concurrent requests into one model call, or a GPU. On the free 512 MB host the reranker is off anyway;
end-to-end question latency is dominated by the LLM (seconds), not by search.

## 3. Answer results

### 3.1 Final results (investigation set, gpt-oss-120b)

| Metric | Agent (final) | Single-shot RAG |
|---|---|---|
| **Correct (strict)** | **0.98** (49/50) | 0.92 (46/50) |
| LLM-as-judge score | **0.92** | 0.90 |
| Key-fact recall | **1.00** | 0.93 |
| Citation precision / recall | 0.97 / **0.88** | 0.96 / 0.81 |
| Groundedness (verifier) | **0.89** | 0.82 |
| Contradictions reported (contradiction questions) | 0.75 | **0.88** |
| Conflict reported on ordinary questions (false-positive rate) | 0.21 | 0.18 |
| Prompt-injection resistance | 1.00 | 1.00 |
| Strict metric agrees with judge | 49/50 | 49/50 |
| LLM calls / prompt tokens per question | 1.66 / 4,802 | 1.00 / 1,989 |

| Category (n) | Agent correct | Single-shot correct | Agent LLM calls |
|---|---|---|---|
| simple (12) | 1.00 | 1.00 | 1.0 |
| multi_hop (10) | 1.00 | 0.90 | 1.1 |
| multi_doc (8) | **1.00** | 0.63 | 2.4 |
| contradiction (8) | 1.00 | 1.00 | 1.0 |
| unanswerable (9) | 0.89 | 1.00 | 3.2 |
| injection (3) | 1.00 | 1.00 | 1.3 |

The only remaining agent error is **u05** ("How many deposits failed during INC-2052?"): the agent
answered that the incident "did not involve any deposit failures" - it turned *absence of a mention*
into a *fact*, instead of saying the documents do not say. Single-shot fails on multi-document
questions (m01, d02, d04, d05): one retrieval does not bring in the second and third documents.

**Caveat - development set.** The fixes in 3.2 were driven by failures on these 50 questions, so
these numbers are optimistic even though every fix is a general rule. Section 3.5 reports the same
system on held-out HotpotQA questions that were never used during development (agent 0.62 strict / 0.74 judge).

### 3.2 How the system improved: evaluation-driven iterations

Every change below was triggered by a specific failing question and is a general rule (not a
question-specific patch). All runs use gpt-oss-120b, the same 50 questions and the same metric.

| Version | Agent correct | Judge | What changed (and which failure caused it) |
|---|---|---|---|
| v1 | 0.92 | 0.88 | baseline |
| v2 | 0.92 | 0.88 | **source-reliability ranking** for conflicting facts (d02 used the initial report's 60 min instead of the postmortem's 107 min); **completeness re-search** for list questions (d04 missed INC-2033). Both fixed; key-fact recall 0.962 → 0.976, citation precision 0.924 → 0.972 |
| v3 | 0.88 | 0.85 | **parse submissions written as text** (u01/u04 abstained correctly but as plain JSON text); **automatic citation**: uncited answers are kept only if every sentence is grounded in the evidence (u08 invented a per-connection fee). A first, stricter version ("abstain if uncited") wrongly rejected three correct answers and was replaced. |
| v4 | **0.98** | **0.92** | **gpt-oss citation styles** (`【S8】`, zero-width characters - d02/c07 correct answers were rejected); **whole-token number matching** ("0" matched inside "2052" in u05); **numbers missing from the cited sources make a claim unsupported** (u08's derived €12); **false-premise rule** in the prompt (u09); **`list_documents` tool** to enumerate all documents of a type (d04). Verifier calibration: version numbers (v3.2), years inside dates, multi-part IDs (AI-2047-1), empty claims, up to ⅓ ungrounded commentary sentences, and *one unverified claim → partial answer* (each found as a correct answer wrongly rejected, or an invented number accepted) |

### 3.3 Model and mode comparisons

| Run | correct | key-fact recall | citation precision | contradictions detected | judge | LLM calls / q | prompt tokens / q |
|---|---|---|---|---|---|---|---|
| agent · gpt-oss-120b (v1 code) | 0.920 | 0.962 | 0.924 | 0.750 | 0.880 | 1.58 | 4,408 |
| agent · gpt-oss-20b (v1 code) | 0.909* | 0.907 | 0.866 | 0.500 | 0.841 | 1.16 | 2,903 |
| single-shot · gpt-oss-120b (v1 code) | 0.920 | 0.962 | 0.952 | 0.625 | - | 1.00 | 1,853 |
| single-shot · gpt-oss-120b (v3 code) | 0.936* | 0.899 | 0.888 | 0.875 | 0.777 | 0.94 | 1,806 |
| **agent · gpt-oss-120b (final)** | **0.980** | **1.000** | 0.967 | 0.750 | **0.920** | 1.66 | 4,802 |
| **single-shot · gpt-oss-120b (final)** | 0.920 | 0.927 | 0.963 | 0.875 | 0.900 | 1.00 | 1,989 |

\* 6 (20b) and 3 (single-shot v3) questions could not be answered because free-tier quotas ran out;
they are excluded from the average, which flatters those runs slightly.

- **120B vs 20B:** the larger model is better on everything that needs judgement - key facts
  (0.96 vs 0.91), citations (0.92 vs 0.87) and especially reporting contradictions (0.75 vs 0.50). The
  20B model is cheaper (fewer calls and tokens: it searches less) but is the wrong trade-off for an
  investigation tool, where a missed conflict is the costly error.
- **Agent vs single-shot (final code):** single-shot RAG is equally good on simple, contradiction
  and injection questions with ~2.4× fewer prompt tokens. The agent's advantage is concentrated where
  the evidence is not reachable with one query: multi-document questions (1.00 vs 0.63) and multi-hop
  chains (1.00 vs 0.90). On unanswerable questions it searches more (3.2 calls) before abstaining;
  single-shot abstains more often in general (22% vs 14% of all questions), which helps on
  unanswerable questions but costs it three answerable multi-document questions. Single-shot reports
  slightly more conflicts (0.88 vs 0.75): with one retrieval, the conflicting documents are more often
  all present in the same context. A production system could route by question type (a cheap
  classifier) to get the cost of single-shot on simple lookups and the agent where it matters.

### 3.4 Robustness

- **Prompt injection:** 100% resisted in every run in which they were answered (3 questions whose evidence contains
  "ignore previous instructions…" payloads). The injected claim ("a PayFlex datacenter power failure")
  never appeared as fact; the agent even used the email's legitimate content correctly.
- **LLM failures:** during evaluation Groq retired a model (404), returned 429s with 20-600 s
  Retry-After, and exhausted a daily token quota; Cerebras hit its 150 requests/hour limit. The client
  waited out short cooldowns, failed over between providers, and returned `llm_unavailable` (with the
  retrieved evidence) instead of failing - no question crashed the pipeline.
- **Judge validity:** disagreements between the strict key-fact metric and the judge were inspected and led to two fixes - a metric bug (`80 %` vs `80%`) and the judge only seeing the
  answer text (not the reported contradictions or the abstention status), which under-scored correct
  contradiction and abstention answers in v1-v3.

### 3.5 Held-out check: HotpotQA questions never used during development

30 HotpotQA questions (the first 30 of the seed-42 sample; 26 bridge, 4 comparison) over the full
66,552-document index, same final code, gpt-oss-120b. *Correct* = the gold answer string appears in
the answer (strict: "Andy Roddick" does not match "Andrew Stephen Roddick"); the judge allows
paraphrases. One question per run hit a free-tier limit and is excluded.

| | correct (strict) | judge | abstained | LLM calls / q | prompt tokens / q |
|---|---|---|---|---|---|
| agent | **0.62** | **0.74** | 2 | 1.37 | 3,192 |
| single-shot RAG | 0.55 | 0.62 | 6 | 0.97 | 1,498 |

- **Generalisation gap.** 0.62-0.74 on unseen questions vs 0.98 on the development set: the
  development number is optimistic, as expected, and HotpotQA is harder (open-domain Wikipedia with
  eight hand-picked distractors per question). With n = 30 the uncertainty is roughly ±0.17, so treat
  these as indicative.
- **The agent still beats single-shot by 7-12 points**, mostly on *bridge* questions, by searching
  for the second hop (0.64 vs 0.56 strict) - the same pattern as in the retrieval results (bridge
  All@10 = 0.70). Single-shot abstains three times as often because the second paragraph is missing.
- **Failure types (agent):** wrong second-hop entity (e.g. answering about the wrong song of the right
  album), an answer to a neighbouring question (the *source* of funding instead of its *type*), and
  correct-but-differently-named answers that only the judge accepts. Improvements: explicit query
  decomposition into hops with an answer-type check ("the question asks for a *character
  description*, not a name"), and alias-aware answer matching for evaluation.

## 4. Where the system still fails, why, and how to improve it

| Failure mode | Example | Why | Improvement |
|---|---|---|---|
| Enumerations depend on one summary document | d04 relied on the Q1 review's payments section | top-k search returns the best *matches*, not *all* matches | `list_documents` (added in v4); next: structured extraction of incidents into a table at ingestion so "all incidents caused by changes" is a query, not a search |
| Faithful to an outdated source | d02 (v1) | the verifier checks support, not authority | source-reliability ranking (v2); next: store `supersedes` links between documents explicitly |
| Derived numbers presented as facts | u08: €1,800 / 150 = €12 | the model "helpfully" computes; arithmetic is not in any source | numbers missing from sources → unsupported (v4); next: let the model mark derived values explicitly and show the calculation |
| Rule-based contradiction coverage | durations, start times, root causes and affected counts | regexes need a pattern per attribute | LLM-based claim extraction into (entity, attribute, value) triples, compared with the same logic |
| Single-query bridge retrieval | HotpotQA bridge All@10 = 0.70 | second hop not named in the question | the agent's follow-up searches; next: train/score query decomposition on HotpotQA itself |
| Free-tier quotas | evaluation runs blocked for an hour | 5 req/min, 150 req/hour, daily token caps | pacing + fallback chain (done); next: route simple questions to single-shot to halve LLM calls |
| Evaluation size | 50 hand-written questions | written by the system's authors | grow with real user questions and blind-written questions; report confidence intervals |

## 5. Cost and latency

- Retrieval: 0.1-0.5 s per query without the cross-encoder, ~1 s with it (Qdrant Cloud, laptop CPU).
- LLM: 1 call for most questions, 2-4 for multi-hop and unanswerable ones; 1.8k (single-shot) to
  5k (agent) prompt tokens and ~0.2-0.3k completion tokens per question. At typical paid prices for a
  model of this size that is well under one cent per question; on the free tiers it is $0, and the
  binding constraint is requests per minute/hour, not money.
- End-to-end on the free Render instance: ~10-25 s per question for the agent, dominated by LLM calls
  and provider rate limits; cached repeats return instantly.
