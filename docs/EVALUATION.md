# Evaluation report

All numbers in this report are produced by `eval/run_eval.py` and stored in `eval/results/*.json`
(per-question details + aggregates + git commit). `python -m eval.report` regenerates the tables in
[eval_tables.md](eval_tables.md). Everything is reproducible: fixed corpora, fixed question sets,
fixed random seeds, temperature 0, and an LLM response cache.

## 1. Methodology

### Question sets

| Set | Size | Ground truth | Used for |
|---|---|---|---|
| Acme Play investigation questions (`eval/investigation_questions.py`) | 49 | hand-written from the corpus generator's ground truth | retrieval + answers |
| HotpotQA distractor sample (`eval/datasets/hotpotqa_questions.jsonl`, seed 42) | 300 | the dataset's gold supporting paragraphs and answers | retrieval (300), answers (subset) |

Investigation questions by category:

| Category | n | What a correct system must do |
|---|---|---|
| simple | 12 | answer from one document |
| multi_hop | 10 | chain facts across documents (incident → change → author/approval/owner) |
| multi_doc | 8 | aggregate or compare several documents (incl. the task's example question) |
| contradiction | 8 | give the reliable answer **and** report the conflicting sources |
| unanswerable | 9 | abstain (8) or correct a false premise (1) - never invent |
| injection | 3 | answer correctly although the evidence contains prompt-injection text |

HotpotQA questions are `bridge` (two-hop chain) or `comparison` (two entities); every one needs two
gold paragraphs hidden among 66,581 documents, including 8 hand-picked distractors per question.

### Retrieval metrics (document level)

Evidence is expressed as **groups** of interchangeable documents (e.g. "the postmortem *or* its
draft"). For the top-k distinct documents returned:
- **Recall@k** - share of evidence groups with at least one document in the top k.
- **All@k** - 1 if *every* group is covered (complete evidence; what multi-hop answers need).
- **MRR** - 1 / rank of the first relevant document.

### Answer metrics

- **Correct** - answerable: all key facts present (strict) and no abstention; unanswerable: abstained;
  false premise: abstained or premise corrected; HotpotQA: gold answer contained in the answer.
- **Key-fact recall** - share of required facts present (partial credit).
- **Citation precision** - share of cited documents that are gold or acceptable evidence.
- **Citation recall** - share of evidence groups covered by the citations.
- **Groundedness** - share of claims the verifier marks as supported (partial = 0.5).
- **Contradiction detection** - share of contradiction questions where a conflict is reported;
  **false-positive rate** - conflicts reported on ordinary answerable questions.
- **Injection resistance** - share of injection-exposed questions whose answer contains none of the
  injected payload strings.
- **Cost / latency** - LLM calls, prompt and completion tokens, wall time per question.

### Configurations compared

1. Retrieval: dense only · BM25 only · hybrid RRF · hybrid + cross-encoder · hybrid + cross-encoder + MMR.
2. Chunking: structure-aware vs fixed 220-word windows (same questions, separate indexes).
3. Answering: agent (multi-step tool calling) vs single-shot RAG (one retrieval, one LLM call).
4. LLMs: two different models on the same questions.

_Results and analysis follow below._
