# AI usage report

I built this project with Claude Code (Anthropic) as a pair programmer: it wrote most of the code,
tests and documentation drafts under my direction, and it ran the evaluations with me. This report
covers my own role, how the AI was used, and the problems we found by running the system for real.

## My role

- **Owned the direction and scope.** I chose the overall approach and the datasets, and set the
  priorities: all mandatory requirements first, bonuses after.
- **Ran all infrastructure and secrets.** I created and configured the GitHub repository (including
  the CI workflow permission), the Qdrant Cloud cluster, the Groq, Cerebras and Google AI Studio API
  keys, and the Render service. I kept every secret myself, in my local `.env` and in Render's
  dashboard; the AI assistant never saw or handled the keys.
- **Deployed it.** When Hugging Face Spaces turned out to require a paid plan, we moved to Render. I
  created the service from the Blueprint, set the secrets and verified the live deployment.
- **Held the quality bar.** I audited the finished work against every requirement and bonus in the
  brief, refused to leave the two missing bonus items (embedding-model comparison, load test) undone,
  and required a full re-check before submission: tests, CI, a secret scan of the git history, the
  documentation, and the live demo end to end.
- **Added XML support.** I implemented `.xml` as the ninth input format by completing the
  `load_xml` loader from a skeleton the assistant provided: the parse-error handling, which elements
  count as text, and how the paragraphs are joined. The assistant integrated it and added the test.
- **Added search filters to the API.** I extended `GET /api/search` with a document-type filter and
  a validated date range, passed through to the retriever's Qdrant filters (completed from a skeleton;
  the assistant caught one bug in my version, returning the raw result object instead of JSON).
- **Extended the contradiction detector.** I added a fourth conflict type, affected counts (e.g.
  delayed transactions), completing the extraction loop from a skeleton the assistant provided: the
  number group, removing thousands separators so 41,300 and 41300 are equal, and recording the fact.
- **Own the code.** I am going through the codebase module by module (ingestion, indexing, hybrid
  retrieval, agent, verification, evaluation) so that I can explain every design decision and change
  the code without AI assistance.

## How AI was used

Claude Code translated and analysed the Georgian brief, proposed the architecture and the two-corpus
design, wrote the first versions of all modules, the tests, the corpus generator and the evaluation
harness, and drafted the documentation. Everything was then run against real data, real LLMs and real
free-tier limits; the evaluation results drove most of the changes.

## Problems found by running the system

None of these showed up in unit tests; they appeared only with real data, real models and real limits.

| Problem | How it showed up | Fix |
|---|---|---|
| Embedding 66k paragraphs took over an hour | profiling: one long paragraph padded a whole batch | sort texts by length before batching (2× faster, identical vectors) |
| LLM providers retired models and rate-limited us | 404s, 429s with long Retry-After, a daily quota exhausted mid-run | configurable models, waiting out short cooldowns, provider fallback chain |
| Agent used a superseded number (initial report's 60 min instead of the postmortem's 107) | evaluation failure | source-reliability ranking for conflicting facts |
| Agent missed one incident in a "list all" question | evaluation failure | completeness re-search and a `list_documents` tool |
| Model invented a fee by arithmetic, with no citation | evaluation failure | numbers missing from every cited source make a claim unsupported |
| A first "abstain if uncited" rule rejected three correct answers | final evaluation run | post-hoc attribution of each sentence to the evidence |
| Metric bug: `80 %` did not match `80%` | one question failed in every run | canonicalized matching, runs re-scored offline |
| Hugging Face Spaces now requires a paid plan for Docker | `402 Payment Required` at deploy | Render free tier; measured RAM (650 MB → 380 MB) and turned the reranker off in production |
