"""Hybrid retriever: the full search pipeline in one place.

    query ─┬─ dense (bge-small, cosine)  ─┐
           └─ BM25 (our sparse vectors) ──┴─ weighted RRF ─ cross-encoder rerank ─ MMR ─ top-k

Every stage can be switched off through RetrievalConfig, which is how the evaluation compares
configurations (dense-only, BM25-only, hybrid, hybrid + rerank, hybrid + rerank + MMR).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from app.config import Settings
from app.index.bm25 import BM25Encoder
from app.index.embeddings import Embedder
from app.index.store import VectorStore, build_filter
from app.models import SearchHit
from app.retrieval.fusion import weighted_rrf
from app.retrieval.mmr import mmr_select
from app.retrieval.rerank import CrossEncoderReranker


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # "dense" | "sparse" | "hybrid"
    candidate_k: int = 40
    final_k: int = 6
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    use_reranker: bool = True
    rerank_top_n: int = 30
    use_mmr: bool = True
    mmr_lambda: float = 0.75

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            candidate_k=s.candidate_k, final_k=s.final_k, rrf_k=s.rrf_k, dense_weight=s.dense_weight,
            sparse_weight=s.sparse_weight, use_reranker=s.use_reranker, rerank_top_n=s.rerank_top_n,
            use_mmr=s.use_mmr, mmr_lambda=s.mmr_lambda,
        )

    def with_overrides(self, **kwargs) -> RetrievalConfig:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    best_rerank_score: float | None = None


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25Encoder,
                 reranker: CrossEncoderReranker | None, config: RetrievalConfig):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config

    def search(
        self,
        query: str,
        corpus: str | None = None,
        doc_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = config or self.config
        flt = build_filter(corpus=corpus, doc_types=doc_types, date_from=date_from, date_to=date_to)
        timings: dict[str, float] = {}

        lists, weights = [], []
        if cfg.mode in ("dense", "hybrid"):
            t = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            lists.append(self.store.search_dense(qvec, cfg.candidate_k, flt))
            weights.append(cfg.dense_weight)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if cfg.mode in ("sparse", "hybrid"):
            t = time.perf_counter()
            lists.append(self.store.search_sparse(self.bm25.encode_query(query), cfg.candidate_k, flt))
            weights.append(cfg.sparse_weight)
            timings["sparse_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = weighted_rrf(lists, weights, k=cfg.rrf_k) if len(lists) > 1 else lists[0]
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000

        best_rerank = None
        if cfg.use_reranker and self.reranker is not None and hits:
            t = time.perf_counter()
            hits = self.reranker.rerank(query, hits[: cfg.rerank_top_n])
            best_rerank = hits[0].scores.get("rerank")
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        if cfg.use_mmr and len(hits) > cfg.final_k:
            t = time.perf_counter()
            missing = [h.chunk.chunk_id for h in hits if h.vector is None]
            if missing:  # BM25-only hits come without vectors; fetch them in one call
                vectors = self.store.get_dense_vectors(missing)
                for h in hits:
                    if h.vector is None:
                        h.vector = vectors.get(h.chunk.chunk_id)
            hits = mmr_select(hits, cfg.final_k, cfg.mmr_lambda)
            timings["mmr_ms"] = (time.perf_counter() - t) * 1000
        else:
            hits = hits[: cfg.final_k]
        return RetrievalResult(hits=hits, timings_ms=timings, best_rerank_score=best_rerank)
