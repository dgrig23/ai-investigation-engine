# Architecture and technical decisions

This document explains how the system works and *why* each choice was made. Code references are
given as `file:function` so every statement can be checked in the source.

## 1. The big picture

```
                         INGESTION                                          QUESTION ANSWERING
 file (pdf/docx/md/html/txt/csv/json)                       question
   │ loaders.load_file          text + format metadata         │ retriever.search (step 0, before any LLM call)
   │ metadata.build_metadata    entities, dates, doc_type      ▼
   │ dedup.content_hash         exact duplicate? -> stop     LLM agent loop (investigator.stream)
   │ dedup.minhash + LSH        near-duplicate? -> link         │  tools: search_documents · find_mentions · read_document
   │ chunking.*Chunker          fixed | structure               │  every result -> numbered evidence [S1], [S2]…
   │ embeddings.Embedder        dense 384-d (bge-small)         │  conflict hints from contradictions.find_conflicts
   │ bm25.BM25Encoder           sparse BM25 weights             ▼
   ▼                                                         submit_answer (structured JSON via tool call)
 Qdrant: chunks_{strategy} (dense + sparse + payload)           │
         documents_{strategy} (registry: versions, hashes)      ▼
                                                             verify.verify_answer
                                                                citations exist? numbers/IDs in sources?
                                                                support score · abstention gate · conflicts merged
                                                                │
                                                                ▼
                                                             answer + claims + contradictions + sources + trace
```

Layers and their single responsibility:

| Package | Responsibility | Talks to |
|---|---|---|
| `app/ingestion` | parse files, extract metadata, dedup, chunk, orchestrate add/update/delete | index |
| `app/index` | embeddings, BM25 encoding, **all** Qdrant access (`store.py` is the only DB module) | Qdrant |
| `app/retrieval` | hybrid search pipeline: fusion, rerank, MMR | index |
| `app/llm` | one provider-agnostic LLM client with retries, fallback, cache | LLM APIs |
| `app/agent` | agent loop, tools, evidence registry, verification, contradictions, injection defence | retrieval, llm |
| `app/api` | HTTP API, streaming, rate limiting, UI | services |
| `app/services.py` | builds and shares components (models load once) | everything |
| `eval/` | metrics, question sets, runners, report | services |

## 2. Data

**Two corpora, two purposes** (decision: use a benchmark for scale and measurement, and a
synthetic corpus for the investigation scenario the task describes):

1. **HotpotQA** (distractor, validation split, 7,405 questions). Each question ships 10 Wikipedia
   paragraphs (2 gold + 8 hard distractors). Pooling them gives 73,700 paragraphs (a few questions have fewer than 10); dedup leaves
   66,552 unique documents (67,680 chunks) indexed. Gold supporting
   paragraphs give exact ground truth for Recall@k/MRR on multi-hop (`bridge`) and comparison questions.
   A fixed random sample (seed 42) is used for evaluation - fully reproducible.
2. **Acme Play incident corpus** (`scripts/generate_incident_corpus.py`): 55 files in 7 formats for
   a fictional gaming platform. Generated from one ground truth, so every answer is known. It
   contains planted cause chains, contradictions, superseded facts, an exact duplicate, a near-duplicate
   draft, missing information and prompt-injection text. This is the corpus that reproduces the task's
   example question ("which changes caused the March payment delays…").

## 3. Ingestion

- **Loaders** (`loaders.py`) normalize every format to "light markdown" (headings as `#`, blank-line
  paragraphs). One shared shape lets one structure-aware chunker serve all formats. Format metadata is
  kept: PDF info dict, DOCX core properties, HTML `<meta>`, Markdown front matter.
- **Metadata** (`metadata.py`) is rule-based (regex) - zero LLM cost, deterministic, fast on 66k docs.
  Entity IDs (`INC-2041`, `CHG-1042`) and service names are stored on every chunk and indexed, which
  powers the `find_mentions` tool: cross-document linking without building a separate graph DB.
- **Dedup** (`dedup.py`): exact = SHA-256 of normalized words (formatting-insensitive). Near-dup =
  MinHash (128 permutations, vectorized numpy) + LSH (16 bands × 8 rows, threshold ≈ 0.71), confirmed
  with exact Jaccard ≥ 0.8. *Decision:* exact duplicates are not indexed; near-duplicates are indexed
  but linked, because a revised report that changes one number is evidence, not noise.
- **Chunking** (`chunking.py`), compared in evaluation:
  - `FixedWindowChunker`: 220 words, 40 overlap. Baseline. Breaks sentences, rows and sections.
  - `StructureChunker`: splits on headings, packs whole paragraphs up to 220 words, splits long
    paragraphs at sentence boundaries, never splits a CSV/JSON record, carries the section name.
  Each chunk is embedded with a context header (`title > section`), which helps short chunks like
  "Root cause: …" keep their meaning.
- **Updates** (`pipeline.py:ingest_document`): content hash unchanged → no-op (idempotent). Otherwise a
  new version is written first, then the old version's chunks are deleted ("write new, then delete
  old"), so search never sees a half-updated document.

## 4. Retrieval (`retriever.py`)

```
query ─┬─ dense: bge-small-en-v1.5 (ONNX, CPU), cosine, top 40 ─┐
       └─ sparse: our BM25 → Qdrant sparse vectors (IDF server-side), top 40 ─┴─ weighted RRF ─ cross-encoder (top 30) ─ MMR ─ top k
```

- **Why hybrid:** dense search understands paraphrase ("deposits were slow" ≈ "payment delays");
  BM25 is exact on identifiers ("CHG-1042", "idx_entries_account_created") where embeddings are weak.
- **BM25 implementation (ours):** tokenizer keeps identifiers whole plus their parts, light stemmer,
  TF saturation and length normalization computed at index time; Qdrant applies IDF with
  `Modifier.IDF`, so IDF stays correct when documents are added/deleted. Hashing trick for term ids.
- **Weighted RRF (ours):** rank-based, so incomparable score scales (cosine vs BM25) do not matter.
- **Cross-encoder re-ranking:** ms-marco-MiniLM-L-6-v2 on the top 30 - reads query and passage
  together, much more precise than the bi-encoder; its best score is the "retrieval confidence" used
  by the abstention gate.
- **MMR (ours):** diversity so multi-document questions are not answered from five chunks of one file.
- **Filters:** corpus, doc type, date range, entity - all Qdrant payload filters (indexed in the cloud).

## 5. Agentic RAG (`investigator.py`)

- Retrieval runs **before** the first LLM call. Simple questions then need exactly one LLM call; only
  multi-hop questions pay for more. (Cost decision, important on free tiers.)
- Tool calling (OpenAI-compatible function calling, works on Groq, Cerebras, Mistral, Gemini):
  `search_documents(query, doc_type?, date_from?, date_to?)`, `find_mentions(entity)`,
  `read_document(doc_id)`, `list_documents(doc_type)` (enumerates every document of a type, for "all X" questions where top-k search cannot guarantee completeness), and `submit_answer(...)` - the final answer itself is a tool call, which
  forces a structured result (answer, claims with citations, contradictions, status, missing info).
- Step budget (default 4) and a context budget (24k chars); on the last step only `submit_answer` is
  offered with `tool_choice="required"`.
- `mode="single_shot"` = classic RAG baseline (one retrieval, one LLM call) for comparison.

## 6. Grounding, contradictions, abstention, security

- **Evidence registry** (`evidence.py`): every chunk shown to the model gets an id; only those ids are
  valid citations.
- **Verification** (`verify.py`), no extra LLM call: removes invented citations; checks that every
  number/time/ID in a claim literally appears in the cited chunks (the most harmful hallucination is
  a wrong number attached to a real source); lexical overlap + embedding similarity give a support label.
- **Abstention:** the prompt requires `insufficient_evidence` when evidence is missing; the verifier
  additionally abstains when no claim is supported, or when retrieval confidence is low *and* support
  is weak.
- **Contradictions** (`contradictions.py`): rule-based extraction of (incident, attribute, value) facts -
  duration, start time, root cause - compared across *different* documents. Conflicts are shown to the
  model during the investigation and merged into the answer if the model omitted them. The model is
  told how to weigh sources (final postmortem > initial report > chat; newer > older).
- **Prompt injection** (`security.py`): documents are untrusted. Pattern detection at ingestion and
  query time, `<document>` isolation, explicit warnings on flagged chunks, role-marker defanging.
  Evaluated by questions whose evidence contains injection payloads.

## 7. LLM client (`app/llm/client.py`)

- Providers: Groq, Cerebras, Mistral, Gemini - all free tiers, all through the OpenAI SDK with a
  different `base_url`. Order in `LLM_PROVIDERS` = fallback order.
- Failure handling: exponential backoff + jitter for timeouts/5xx; `Retry-After` honoured for short
  429s, otherwise fail over; circuit breaker per provider; auth/model errors open the circuit for an
  hour; forced `tool_choice` downgraded to `auto` if a provider rejects it; lenient JSON repair for
  tool arguments; portable tool-call ids (Mistral requires 9 alphanumeric characters).
- If **all** providers fail the agent returns the retrieved evidence instead of an error
  (`status: llm_unavailable`) - the user can still investigate.
- SQLite response cache keyed by (model, messages, tools): identical calls are free, evaluation re-runs
  are reproducible.

## 8. Why these technologies

| Decision | Alternatives considered | Reason |
|---|---|---|
| Qdrant | pgvector (Neon/Supabase), Chroma, Pinecone | Native dense + sparse vectors in one collection, server-side IDF, payload filters/indexes, delete-by-filter for updates; free 1 GB cloud cluster; **embedded local mode** makes tests and evaluation run with no server. pgvector would give SQL transactions, but hybrid search needs a separate FTS setup and the free tiers have 0.5 GB storage. |
| bge-small-en-v1.5 via fastembed (ONNX) | OpenAI/Gemini embedding APIs, bge-base | Free, no rate limits for 66k docs, same model in the container, 384-d keeps the index small for the free tier. |
| No LangChain / LlamaIndex | both | The pipeline is the product; owning every step makes it explainable, testable and debuggable, and the task requires our own algorithms. |
| FastAPI + one HTML page | Streamlit, Gradio | One process, one container, native streaming (SSE), the API is usable without the UI. |
| Render free web service (Docker) | Hugging Face Spaces, Koyeb, Vercel | Hugging Face now requires a paid PRO plan for Docker/Gradio Spaces (only static Spaces are free - discovered when deploying). Render deploys the Dockerfile straight from GitHub for free; its 512 MB RAM forced a measured trade-off: the cross-encoder is disabled in production (peak RAM 650 MB → 380 MB). Vercel's serverless functions cannot keep ONNX models loaded. |

## 9. Performance and cost

- Retrieval latency is dominated by the cross-encoder (~tens of ms on CPU for 30 pairs) and the
  network round trip to Qdrant Cloud; see the evaluation report for measured p50 values.
- LLM cost per question: 1 call for simple questions, 2-4 for multi-hop; prompts are kept compact
  (700-char snippets, 6 results per search). Measured tokens per question are in the evaluation report.
  On free tiers the cost is $0; the daily request/token quotas are the real limit, handled by
  fallback, caching and the per-IP rate limit on the public demo.
- Caches: query-embedding LRU, final-answer LRU (API), LLM response cache (SQLite).
