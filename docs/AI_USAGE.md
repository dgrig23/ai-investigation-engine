# AI usage report

> Draft prepared during development. **Rewrite the last section in your own words before submitting**
> - the task asks what *you* did independently, and only you can answer that honestly.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic Claude, Claude desktop app) | Translating and analysing the Georgian task brief, proposing the architecture, writing most of the implementation, tests, corpus generator, evaluation harness and documentation drafts; running the evaluations and debugging. |
| _add any other tools you used_ | |

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

These were not visible in unit tests - they appeared only with real data, real models and real limits.

| Problem | Evidence | Fix |
|---|---|---|
| Document registry shared between chunking strategies | indexing the 2nd strategy would mark every file "unchanged" | registry per strategy |
| Contradiction detector attributed every number in a chunk to its first incident | false conflict from a weekly report listing two incidents | attribute facts to the nearest preceding incident ID; test added |
| Bulk dedup was O(n²) | 74k paragraphs | set lookup |
| Embedding 66k paragraphs took > 1 hour | profiling: one 423-token paragraph padded a whole batch | sort by length before batching (2× faster); verified identical vectors |
| Reranker took 2.3 s per query | benchmark | rerank top 30 with 800-char passages (0.3-0.7 s) |
| Groq retired `llama-3.3-70b`, Gemini's free tier dropped `gemini-2.5-flash` | 404 on first real call (circuit breaker opened correctly) | models configurable; gpt-oss-120b, gemini-flash-lite-latest |
| Free-tier 429s with 20-30 s Retry-After failed whole questions | smoke test | wait for the shortest provider cooldown (≤ 60 s) before giving up |
| Groq's daily token quota ran out mid-evaluation | instant "unavailable" responses | fallback chain Groq → Cerebras (same model) - the failure handling working as designed |
| Model writes `CHG‑1042` with a non-breaking hyphen | correct answers scored wrong; ID checks failed | normalize Unicode in model output |
| Metric bug: `80 %` did not match `80%` | one question failed in every run | canonicalize before matching; runs re-scored offline |
| Agent used a superseded value (initial report's 60 min instead of postmortem's 107) | evaluation failure d02 | source-reliability ranking in conflict hints |
| Agent stopped after one search on a list question | evaluation failure d04 | completeness re-search instruction |
| Model wrote the final JSON as text instead of a tool call | correct abstentions scored as answers (u01, u04) | parse text submissions |
| Model invented a per-connection fee by arithmetic, with no citation | evaluation failure u08 | strict grounding: uncited answers abstain |
| Hugging Face Spaces now requires PRO for Docker | `402 Payment Required` at deploy | Render free tier; measured peak RAM 650 MB → reranker off in production (380 MB) |
| Qdrant Cloud queries slow (1.3 s) | per-stage timing | fetch vectors only when MMR needs them (168 ms) |

## What I did / changed independently

_Write this yourself. Examples of what belongs here: decisions you made or changed, parts you
rewrote, bugs you found, experiments you ran, parameters you tuned, accounts and infrastructure you set
up (Qdrant Cloud, Render, API keys, GitHub), what you verified by hand, and which parts of the code you
can explain line by line._
