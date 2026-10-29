# AI usage report

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic Claude, Claude desktop app) | Translating and analysing the Georgian task brief, proposing the architecture, writing most of the implementation, tests, corpus generator, evaluation harness and documentation drafts; running the evaluations and debugging. |

## How AI was used, step by step

1. **Requirements analysis.** The brief was translated into a checklist; three project directions were
   compared and the two-corpus design was chosen (HotpotQA for scale and measurable multi-hop retrieval,
   a synthetic incident corpus for the investigation scenario from the brief).
2. **Architecture.** Component split (ingestion / index / retrieval / llm / agent / api), Qdrant with
   dense + sparse vectors, local ONNX models, an all-free LLM setup with provider fallback.
3. **Implementation.** First versions of all modules were AI-generated, then run against real data,
   real LLMs and real free-tier limits, and corrected. The evaluation drove most changes.
4. **Evaluation.** The question set and metrics were drafted with AI; expected answers were checked
   against the corpus generator's ground truth.

## Problems found by running the system (and how they were fixed)

These were not visible in unit tests: they appeared only with real data, real models and real limits.

| Problem | Evidence | Fix |
|---|---|---|
| Document registry shared between chunking strategies | indexing the 2nd strategy would mark every file "unchanged" | registry per strategy |
| Contradiction detector attributed every number in a chunk to its first incident | false conflict from a weekly report listing two incidents | attribute facts to the nearest preceding incident ID; test added |
| Bulk dedup was O(n²) | 74k paragraphs | set lookup |
| Embedding 66k paragraphs took > 1 hour | profiling: one 423-token paragraph padded a whole batch | sort by length before batching (2× faster); verified identical vectors |
| Reranker took 2.3 s per query | benchmark | rerank top 30 with 800-char passages (0.3-0.7 s) |
| Groq retired `llama-3.3-70b`, Gemini's free tier dropped `gemini-2.5-flash` | 404 on first real call (circuit breaker opened correctly) | models configurable; gpt-oss-120b, gemini-flash-lite-latest |
| Free-tier 429s with 20-30 s Retry-After failed whole questions | smoke test | wait for the shortest provider cooldown (≤ 60 s) before giving up |
| Groq's daily token quota ran out mid-evaluation | instant "unavailable" responses | fallback chain Groq → Cerebras (same model): the failure handling working as designed |
| Model writes `CHG‑1042` with a non-breaking hyphen | correct answers scored wrong; ID checks failed | normalize Unicode in model output |
| Metric bug: `80 %` did not match `80%` | one question failed in every run | canonicalize before matching; runs re-scored offline |
| Agent used a superseded value (initial report's 60 min instead of postmortem's 107) | evaluation failure d02 | source-reliability ranking in conflict hints |
| Agent stopped after one search on a list question | evaluation failure d04 | completeness re-search instruction and a `list_documents` tool |
| Model wrote the final JSON as text instead of a tool call | correct abstentions scored as answers (u01, u04) | parse text submissions |
| Model invented a per-connection fee by arithmetic, with no citation | evaluation failure u08 | uncited answers are attributed sentence by sentence; a number missing from every source makes a claim unsupported |
| A first "abstain if uncited" rule rejected three correct answers | final evaluation run | replaced by post-hoc attribution (see above) |
| Hugging Face Spaces now requires PRO for Docker | `402 Payment Required` at deploy | Render free tier; measured peak RAM 650 MB → reranker off in production (380 MB) |
| Qdrant Cloud queries slow (1.3 s) | per-stage timing | fetch vectors only when MMR needs them (168 ms) |

## What I did / changed independently

I led the project and made its decisions; most of the code was written by Claude Code under my
direction, as described above. My own part:

- **Direction and scope.** I chose the project direction from the options we discussed: HotpotQA for
  scale and measurable retrieval, plus a synthetic incident corpus that reproduces the brief's
  payment-delay example. With two days until the deadline I set the priority order: every mandatory
  requirement first, bonuses after.
- **The zero-cost constraint.** The first plan suggested a small paid LLM budget. I pointed out that
  section 4 requires the project to run without paid infrastructure, which changed the design to the
  all-free provider chain (Groq, Cerebras, Gemini) with rate-limit handling and fallback.
- **Infrastructure and secrets.** I created and configured every account the system depends on:
  the GitHub repository (including the permission for the CI workflow), the Qdrant Cloud cluster, the
  Groq, Cerebras and Google AI Studio API keys, and the Render service. I managed all secrets myself,
  in my local `.env` and in the Render dashboard; the AI assistant never saw or entered them.
- **Deployment.** When Hugging Face Spaces turned out to require a paid plan, we moved to Render. I
  created the service from the Blueprint, entered the secrets, and checked that the build was live
  before testing the public URL.
- **Completeness.** When reviewing the finished work against the brief, I insisted that the two
  bonus items still missing (embedding-model comparison and a load test) be done rather than skipped.
- **Ownership and understanding.** I review the code module by module (ingestion → indexing →
  hybrid retrieval → agent → verification → evaluation) with a written study guide, so that I can
  explain each design decision, trace the RAG pipeline step by step and modify the code without AI
  assistance.
