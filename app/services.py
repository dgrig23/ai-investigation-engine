"""Wiring: builds every component once and shares it (a tiny dependency-injection container).

Models are loaded lazily and only once per process, because loading ONNX models takes seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

from app.agent.investigator import AgentConfig, Investigator
from app.config import Settings, get_settings
from app.index.bm25 import BM25Encoder
from app.index.embeddings import Embedder
from app.index.store import VectorStore
from app.ingestion.pipeline import IngestionService
from app.llm.client import LLMClient
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.retriever import HybridRetriever, RetrievalConfig

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings

    @cached_property
    def store(self) -> VectorStore:
        store = VectorStore(self.settings)
        store.ensure_collections()
        return store

    @cached_property
    def embedder(self) -> Embedder:
        log.info("loading embedding model %s", self.settings.embedding_model)
        return Embedder(self.settings.embedding_model)

    @cached_property
    def reranker(self) -> CrossEncoderReranker | None:
        if not self.settings.use_reranker:
            return None
        log.info("loading reranker %s", self.settings.reranker_model)
        return CrossEncoderReranker(self.settings.reranker_model)

    @cached_property
    def bm25(self) -> BM25Encoder:
        s = self.settings
        return BM25Encoder(k1=s.bm25_k1, b=s.bm25_b, avgdl=s.bm25_avgdl)

    @cached_property
    def retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig.from_settings(self.settings)

    @cached_property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.store, self.embedder, self.bm25, self.reranker, self.retrieval_config)

    @cached_property
    def ingestion(self) -> IngestionService:
        s = self.settings
        return IngestionService(self.store, self.embedder, self.bm25, s.chunk_strategy, s.chunk_max_words, s.chunk_overlap_words)

    @cached_property
    def llm(self) -> LLMClient:
        return LLMClient(self.settings)

    @cached_property
    def agent_config(self) -> AgentConfig:
        s = self.settings
        return AgentConfig(max_steps=s.agent_max_steps, snippet_chars=s.agent_snippet_chars,
                           max_context_chars=s.agent_max_context_chars, abstain_threshold=s.abstain_rerank_threshold,
                           retrieval=self.retrieval_config)

    @cached_property
    def investigator(self) -> Investigator:
        return Investigator(self.retriever, self.store, self.llm, self.embedder, self.agent_config)


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services(get_settings())
    return _services
