"""Central configuration.

All settings come from environment variables (or a local `.env` file), so secrets never
live in code. Every tunable used by retrieval and the agent is here, which makes it easy
to run evaluation with different configurations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # --- Vector database ---
    qdrant_url: str | None = None  # None/empty -> embedded local mode
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = str(DATA_DIR / "qdrant_local")
    collection_prefix: str = "ie"

    # --- Models (run locally on CPU via ONNX, free) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    onnx_threads: int | None = None  # None = all cores; 1 on tiny free-tier CPUs avoids oversubscription

    # --- Chunking ---
    chunk_strategy: str = "structure"  # "structure" | "fixed"
    chunk_max_words: int = 220
    chunk_overlap_words: int = 40

    # --- BM25 ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_avgdl: float = 90.0  # average chunk length in tokens, measured on the corpus at index time

    # --- Retrieval ---
    candidate_k: int = 40  # candidates pulled from each retriever before fusion
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    use_reranker: bool = True
    rerank_top_n: int = 30
    use_mmr: bool = True
    mmr_lambda: float = 0.75
    final_k: int = 6
    # Below this cross-encoder score the best evidence is considered too weak to answer from.
    abstain_rerank_threshold: float = -4.0

    # --- LLM ---
    llm_providers: str = "groq,cerebras,mistral,gemini"
    groq_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    cerebras_model: str = "gpt-oss-120b"
    mistral_model: str = "mistral-small-latest"
    gemini_model: str = "gemini-flash-lite-latest"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_max_wait_s: float = 60.0  # wait this long at most when every provider is rate-limited
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2000
    # Reasoning models (gpt-oss) think before answering; "low" keeps latency and tokens down.
    llm_reasoning_effort: str | None = "low"
    llm_cache_path: str = str(DATA_DIR / "cache" / "llm_cache.sqlite")

    # --- Agent ---
    agent_max_steps: int = 4
    agent_snippet_chars: int = 700
    agent_max_context_chars: int = 24000

    # --- API ---
    admin_token: SecretStr | None = None
    ask_rate_limit_per_minute: int = 12
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
