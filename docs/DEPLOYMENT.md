# Deployment (100% free tiers)

```
   browser ──HTTPS──► Hugging Face Space (Docker, free CPU: 2 vCPU / 16 GB RAM)
                         FastAPI + UI, ONNX models baked into the image
                         │                         │
                         │ gRPC/HTTPS + API key    │ HTTPS + API keys (fallback chain)
                         ▼                         ▼
                 Qdrant Cloud free cluster     Groq · Cerebras · Mistral · Gemini (free tiers)
                 (1 GB RAM, persistent)
```

- **State lives only in Qdrant Cloud.** The Space's disk is ephemeral; restarting or rebuilding the
  Space loses nothing except caches. Documents uploaded through the UI are written to Qdrant, so they
  survive restarts.
- **Secrets** are Space *secrets* (environment variables), never in the repository: `.env` is
  git-ignored and `.env.example` holds only empty placeholders. Settings are read by
  `pydantic-settings` and keys are `SecretStr`, so they are never printed in logs or reprs.

## Steps

1. **Qdrant Cloud** - create a free cluster, copy its URL and an API key.
2. **Index the data into the cloud from your machine** (embedding runs locally on CPU):
   ```bash
   # in .env: QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333  QDRANT_API_KEY=...
   .venv/bin/python -m scripts.index_corpus --corpus incidents
   .venv/bin/python -m scripts.index_corpus --corpus hotpotqa
   ```
3. **Hugging Face Space** - New Space → SDK *Docker* → hardware *CPU basic (free)*. The README's
   front matter (`sdk: docker`, `app_port: 7860`) configures it. Push the repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
4. **Space secrets** (Settings → Variables and secrets): `QDRANT_URL`, `QDRANT_API_KEY`,
   `GROQ_API_KEY` and/or `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `LLM_PROVIDERS`,
   `ADMIN_TOKEN`.
5. Open the Space URL; `/api/health` shows the vector DB mode (`cloud`) and the active LLM providers.

## Free-tier limits and how the app handles them

| Resource | Limit (free) | Handling |
|---|---|---|
| HF Space | sleeps after inactivity; cold start ≈ 1 min (container start + model load) | models baked into the image; warm-up thread at startup |
| Qdrant Cloud | 1 GB RAM / small disk; idle clusters can be suspended | int8 scalar quantization for vectors, payload on disk; ≈ 66k + 0.3k chunks fit |
| LLM providers | requests/day and tokens/minute quotas, 429s | fallback chain across 4 providers, Retry-After, circuit breaker, response cache, answer cache, per-IP rate limit (`ASK_RATE_LIMIT_PER_MINUTE`) |
| Abuse of the public demo | anyone can call the API | built-in corpora are read-only without `ADMIN_TOKEN`; uploads limited to 10 MB and to supported formats |

Costs: $0. With paid tiers the same design would cost roughly the price of 1-4 LLM calls of
~2-5k input tokens per question (see token counts in the evaluation report).
