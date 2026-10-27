"""Shared test fixtures.

Tests never download models or call real LLMs:
- HashEmbedder: a deterministic bag-of-words embedder (hashing trick) - similar texts get similar
  vectors, which is all retrieval tests need.
- In-memory Qdrant (`QdrantClient(":memory:")`) - the real client code path, no server.
- FakeLLM: returns scripted responses, so agent behaviour is fully reproducible.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.index.bm25 import BM25Encoder, tokenize
from app.index.store import VectorStore
from app.ingestion.pipeline import IngestionService
from app.llm.client import LLMResponse, ToolCall
from app.retrieval.retriever import HybridRetriever, RetrievalConfig


class HashEmbedder:
    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts, batch_size: int = 64) -> np.ndarray:
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """Scripted LLM: each call pops the next response (a ToolCall list or a text answer)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

        class _Stats:
            calls = 0

        self.stats = _Stats()

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        self.stats.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(content=item, tool_calls=[], provider="fake", model="fake")
        calls = [ToolCall(id=f"c{i:08d}", name=n, arguments=a) for i, (n, a) in enumerate(item)]
        return LLMResponse(content="", tool_calls=calls, provider="fake", model="fake", prompt_tokens=100, completion_tokens=20)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, qdrant_url=None, qdrant_path=str(tmp_path / "q"), embedding_dim=HashEmbedder.dim,
                    llm_cache_path=str(tmp_path / "cache.sqlite"), use_reranker=False)


@pytest.fixture
def store(settings) -> VectorStore:
    s = VectorStore(settings, client=QdrantClient(":memory:"))
    s.ensure_collections()
    return s


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def ingestion(store, embedder) -> IngestionService:
    return IngestionService(store, embedder, BM25Encoder(), "structure", max_words=120)


@pytest.fixture
def retriever(store, embedder) -> HybridRetriever:
    cfg = RetrievalConfig(use_reranker=False, use_mmr=True, final_k=5, candidate_k=20)
    return HybridRetriever(store, embedder, BM25Encoder(), None, cfg)
