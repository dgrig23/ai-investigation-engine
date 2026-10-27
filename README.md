---
title: AI Investigation Engine
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AI Investigation Engine

An evidence-based investigation system over large document collections. It answers questions that
need **several documents at once**: it searches, follows links between documents (incident → change →
approval), compares sources, reports **contradictions**, cites every claim, and **refuses to answer**
when the evidence is not there.

> Example (from the task brief): *"Which system changes caused the payment processing delays in
> March, which services were affected, and which documents confirm this?"* - the agent finds the two
> March incidents, links them to CHG-1042 + CHG-1051 and CHG-1047 through the change log and
> postmortems, lists the affected services, cites each fact, and flags that the initial incident
> report blamed the payment provider while the final postmortem proves otherwise.

**Live demo:** _PUBLIC_URL_ · **Evaluation report:** [docs/EVALUATION.md](docs/EVALUATION.md) ·
**Architecture & decisions:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
**Deployment:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) · **AI usage:** [docs/AI_USAGE.md](docs/AI_USAGE.md)

## What it does

| Requirement | How it is met |
|---|---|
| ≥ 3 file formats | PDF, DOCX, Markdown, HTML, TXT, CSV, JSON, JSONL (`app/ingestion/loaders.py`) |
| Text + metadata extraction | per-format metadata (PDF info, DOCX properties, HTML meta, front matter) + rule-based entities, dates, doc type |
| Chunking, 2 strategies compared | fixed sliding window vs structure-aware chunker (`chunking.py`), compared in the evaluation |
| Duplicates | exact (SHA-256) and near-duplicates (MinHash + LSH, implemented from scratch) |
| Add / update / delete | versioned, idempotent ingestion; API + UI |
| Vector DB, semantic + hybrid search | Qdrant: dense (bge-small) + sparse BM25 vectors |
| Own search/ranking algorithms | BM25 encoder, weighted Reciprocal Rank Fusion, MMR diversification, MinHash/LSH, rule-based contradiction detector, claim grounding check |
| RAG, multi-document, multi-step | tool-calling agent: `search_documents`, `find_mentions`, `read_document`, `submit_answer` |
| Citations | every claim cites evidence ids; citations are verified against the sources |
| Contradictions | model + deterministic detector; source reliability guidance |
| Abstention | prompt rules + verifier gate (unsupported claims, low retrieval confidence) |
| LLM errors and limits | retries with backoff, Retry-After, provider fallback, circuit breaker, cache, degraded mode |
| ≥ 50,000 documents | HotpotQA distractor set: 74,050 paragraphs → _N_HOTPOT_ unique documents after dedup |
| Evaluation | 49 investigation questions + 300 HotpotQA questions; Recall@k, All@k, MRR, correctness, citation precision/recall, abstention, contradiction detection, injection resistance; 5 retrieval configs, 2 chunkers, agent vs single-shot, 2 LLMs |
| Python backend + UI | FastAPI + single-page UI with live streamed investigation trace |
| Free deployment | Hugging Face Spaces (Docker) + Qdrant Cloud free tier + free LLM tiers |
| Bonus | agentic RAG, cross-encoder re-ranking, prompt-injection tests, caching, streaming, metrics endpoint, 2 LLMs compared |

## Quick start (local, no accounts needed for search)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env            # add at least one free LLM key (Groq, Cerebras, Mistral or Gemini)

.venv/bin/python -m scripts.generate_incident_corpus      # synthetic Acme Play corpus (~100 files)
.venv/bin/python -m scripts.index_corpus --corpus incidents
.venv/bin/python -m scripts.prepare_hotpotqa              # downloads HotpotQA (27 MB) + builds the eval sample
.venv/bin/python -m scripts.index_corpus --corpus hotpotqa   # embeds ~66k paragraphs on CPU

.venv/bin/python -m uvicorn app.api.main:app --port 8000  # open http://localhost:8000
.venv/bin/python -m pytest                                 # tests: no network, no real models
```

Without `QDRANT_URL` the embedded local Qdrant (`data/qdrant_local`) is used; with it, the same code
talks to Qdrant Cloud.

## Repository layout

```
app/
  ingestion/   loaders.py · metadata.py · chunking.py · dedup.py · pipeline.py
  index/       embeddings.py · bm25.py · store.py (all Qdrant access)
  retrieval/   retriever.py · fusion.py (RRF) · mmr.py · rerank.py
  llm/         client.py (providers, retries, fallback, cache)
  agent/       investigator.py · prompts.py · evidence.py · verify.py · contradictions.py · security.py
  api/         main.py (REST + SSE)        web/index.html (UI)
  config.py · services.py · observability.py
scripts/       generate_incident_corpus.py · prepare_hotpotqa.py · index_corpus.py
eval/          investigation_questions.py · metrics.py · run_eval.py · report.py · results/
tests/         unit + integration tests (in-memory Qdrant, fake embedder, scripted LLM)
docs/          ARCHITECTURE.md · EVALUATION.md · DEPLOYMENT.md · AI_USAGE.md
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status of vector DB and configured LLMs |
| GET | `/api/stats` | documents and chunks per corpus |
| GET | `/api/metrics` | request counts, p50/p95 latency per route, LLM calls and tokens |
| GET | `/api/documents?corpus=` | list documents (paginated) |
| GET | `/api/documents/{doc_id}` | document metadata and chunks |
| POST | `/api/documents` | upload files (multipart `files`, `corpus`) |
| PUT | `/api/documents/{doc_id}` | replace content → new version |
| DELETE | `/api/documents/{doc_id}` | delete document and chunks |
| GET | `/api/search?q=&mode=&rerank=&mmr=` | raw hybrid search with every stage's scores |
| POST | `/api/ask` | `{question, corpus, mode}` → answer, claims, contradictions, sources, trace |
| POST | `/api/ask/stream` | same as Server-Sent Events (live agent steps) |

The built-in corpora (`incidents`, `hotpotqa`) are read-only on the public demo; uploads go to
`uploads` (an `ADMIN_TOKEN` header unlocks the others).

## Evaluation summary

See [docs/EVALUATION.md](docs/EVALUATION.md) for methodology, all tables and the error analysis.

_EVAL_SUMMARY_

## Limitations

- Free tiers: the Space sleeps after inactivity (first request after sleep takes ~1 minute), and LLM
  daily quotas can run out under heavy use - the app then returns the retrieved evidence instead of an
  answer and says so.
- Contradiction detection by rules covers durations, start times and root causes; other conflicts
  rely on the LLM.
- Metadata extraction is rule-based and tuned to identifiers like `INC-1234`; other organizations
  would need their own patterns.
