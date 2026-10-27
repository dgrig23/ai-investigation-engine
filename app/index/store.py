"""Qdrant repository: the only module that talks to the vector database.

Two collections:
- `{prefix}_chunks_{strategy}`: one point per chunk with a named dense vector ("dense"), a named
  sparse BM25 vector ("bm25", IDF applied server-side) and a payload (text + metadata).
  One collection per chunking strategy lets us compare strategies side by side.
- `{prefix}_documents_{strategy}`: the document registry (no vectors). It stores versions, content
  hashes and LSH band keys, which power update/delete and duplicate detection.

The same code runs against embedded local Qdrant (tests, evaluation, development) and Qdrant
Cloud (deployment) - only the connection settings differ.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.index.bm25 import SparseVector
from app.models import Chunk, SearchHit

log = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f1c9a52-3e1b-4c55-9a8e-2b7d0c4e9f10")
DENSE = "dense"
SPARSE = "bm25"


def doc_point_id(doc_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"doc::{doc_id}"))


def chunk_point_id(doc_id: str, version: int, position: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"chunk::{doc_id}::{version}::{position}"))


def date_to_int(date: str | None) -> int | None:
    if not date or len(date) < 10:
        return None
    try:
        return int(date[:10].replace("-", ""))
    except ValueError:
        return None


def build_filter(
    corpus: str | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    entities: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> models.Filter | None:
    must: list[models.Condition] = []
    if corpus:
        must.append(models.FieldCondition(key="corpus", match=models.MatchValue(value=corpus)))
    if doc_types:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=doc_types)))
    if entities:
        must.append(models.FieldCondition(key="entities", match=models.MatchAny(any=entities)))
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
    lo, hi = date_to_int(date_from), date_to_int(date_to)
    if lo or hi:
        must.append(models.FieldCondition(key="date_int", range=models.Range(gte=lo, lte=hi)))
    return models.Filter(must=must) if must else None


def _payload_to_chunk(point_id: str, payload: dict[str, Any]) -> Chunk:
    meta = {k: v for k, v in payload.items() if k not in {"doc_id", "corpus", "text", "position", "section", "title"}}
    return Chunk(
        chunk_id=str(point_id),
        doc_id=payload["doc_id"],
        corpus=payload.get("corpus", ""),
        text=payload.get("text", ""),
        position=int(payload.get("position", 0)),
        section=payload.get("section", ""),
        title=payload.get("title", ""),
        metadata=meta,
    )


class VectorStore:
    def __init__(self, settings: Settings, strategy: str | None = None, client: QdrantClient | None = None):
        self.settings = settings
        self.strategy = strategy or settings.chunk_strategy
        # Each chunking strategy is a fully independent index (its own chunks + its own registry).
        self.chunks_collection = f"{settings.collection_prefix}_chunks_{self.strategy}"
        self.docs_collection = f"{settings.collection_prefix}_documents_{self.strategy}"
        self.client = client or self._connect(settings)
        self.is_local = not settings.qdrant_url

    @staticmethod
    def _connect(settings: Settings) -> QdrantClient:
        if settings.qdrant_url:
            api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=30)
        return QdrantClient(path=settings.qdrant_path)

    # ------------------------------------------------------------------ setup

    def ensure_collections(self) -> None:
        if not self.client.collection_exists(self.chunks_collection):
            self.client.create_collection(
                self.chunks_collection,
                vectors_config={DENSE: models.VectorParams(size=self.settings.embedding_dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
                # int8 quantization keeps the vectors ~4x smaller in RAM (matters on the 1 GB free tier).
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
                ),
                on_disk_payload=True,
            )
            for field, schema in (
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("corpus", models.PayloadSchemaType.KEYWORD),
                ("doc_type", models.PayloadSchemaType.KEYWORD),
                ("entities", models.PayloadSchemaType.KEYWORD),
                ("date_int", models.PayloadSchemaType.INTEGER),
            ):
                self._index(self.chunks_collection, field, schema)
        if not self.client.collection_exists(self.docs_collection):
            self.client.create_collection(self.docs_collection, vectors_config={}, on_disk_payload=True)
            for field in ("doc_id", "corpus", "content_hash", "lsh"):
                self._index(self.docs_collection, field, models.PayloadSchemaType.KEYWORD)

    def _index(self, collection: str, field: str, schema: models.PayloadSchemaType) -> None:
        if self.is_local:
            return  # embedded mode has no payload indexes (it filters by full scan)
        self.client.create_payload_index(collection, field_name=field, field_schema=schema)

    def reset(self) -> None:
        for name in (self.chunks_collection, self.docs_collection):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
        self.ensure_collections()

    # ------------------------------------------------------------------ chunks

    def upsert_chunks(
        self, chunks: list[Chunk], dense: np.ndarray, sparse: list[SparseVector], batch_size: int = 256
    ) -> None:
        for start in range(0, len(chunks), batch_size):
            points = []
            for chunk, vec, sp in zip(
                chunks[start:start + batch_size], dense[start:start + batch_size], sparse[start:start + batch_size],
                strict=True,
            ):
                payload = {
                    "doc_id": chunk.doc_id,
                    "corpus": chunk.corpus,
                    "text": chunk.text,
                    "position": chunk.position,
                    "section": chunk.section,
                    "title": chunk.title,
                    **chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector={DENSE: vec.tolist(), SPARSE: models.SparseVector(indices=sp.indices, values=sp.values)},
                        payload=payload,
                    )
                )
            self.client.upsert(self.chunks_collection, points=points, wait=True)

    def delete_doc_chunks(self, doc_id: str) -> None:
        self.client.delete(
            self.chunks_collection,
            points_selector=models.FilterSelector(filter=build_filter(doc_ids=[doc_id])),
            wait=True,
        )

    def search_dense(self, vector: np.ndarray, limit: int, flt: models.Filter | None = None) -> list[SearchHit]:
        res = self.client.query_points(
            self.chunks_collection, query=vector.tolist(), using=DENSE, limit=limit,
            query_filter=flt, with_payload=True, with_vectors=[DENSE],
        )
        return [
            SearchHit(chunk=_payload_to_chunk(p.id, p.payload or {}), score=float(p.score),
                      scores={"dense": float(p.score)}, vector=(p.vector or {}).get(DENSE))
            for p in res.points
        ]

    def search_sparse(self, sparse: SparseVector, limit: int, flt: models.Filter | None = None) -> list[SearchHit]:
        if not sparse.indices:
            return []
        res = self.client.query_points(
            self.chunks_collection, query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE, limit=limit, query_filter=flt, with_payload=True,
        )
        return [
            SearchHit(chunk=_payload_to_chunk(p.id, p.payload or {}), score=float(p.score), scores={"bm25": float(p.score)})
            for p in res.points
        ]

    def get_dense_vectors(self, chunk_ids: Iterable[str]) -> dict[str, list[float]]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        points = self.client.retrieve(self.chunks_collection, ids=ids, with_payload=False, with_vectors=[DENSE])
        return {str(p.id): (p.vector or {}).get(DENSE) for p in points}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        points = self.client.retrieve(self.chunks_collection, ids=[chunk_id], with_payload=True)
        return _payload_to_chunk(points[0].id, points[0].payload or {}) if points else None

    def get_doc_chunks(self, doc_id: str, limit: int = 500) -> list[Chunk]:
        points, _ = self.client.scroll(
            self.chunks_collection, scroll_filter=build_filter(doc_ids=[doc_id]), limit=limit, with_payload=True
        )
        chunks = [_payload_to_chunk(p.id, p.payload or {}) for p in points]
        return sorted(chunks, key=lambda c: c.position)

    def find_chunks(self, flt: models.Filter, limit: int = 20) -> list[Chunk]:
        points, _ = self.client.scroll(self.chunks_collection, scroll_filter=flt, limit=limit, with_payload=True)
        return [_payload_to_chunk(p.id, p.payload or {}) for p in points]

    def count_chunks(self, corpus: str | None = None) -> int:
        return self.client.count(self.chunks_collection, count_filter=build_filter(corpus=corpus), exact=True).count

    # ------------------------------------------------------------------ documents

    def upsert_document(self, record: dict[str, Any]) -> None:
        self.client.upsert(
            self.docs_collection,
            points=[models.PointStruct(id=doc_point_id(record["doc_id"]), vector={}, payload=record)],
            wait=True,
        )

    def upsert_documents(self, records: list[dict[str, Any]], batch_size: int = 512) -> None:
        for start in range(0, len(records), batch_size):
            points = [models.PointStruct(id=doc_point_id(r["doc_id"]), vector={}, payload=r)
                      for r in records[start:start + batch_size]]
            self.client.upsert(self.docs_collection, points=points, wait=True)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        points = self.client.retrieve(self.docs_collection, ids=[doc_point_id(doc_id)], with_payload=True)
        return dict(points[0].payload or {}) if points else None

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.docs_collection, points_selector=models.PointIdsList(points=[doc_point_id(doc_id)]), wait=True)

    def _find_docs(self, conditions: list[models.Condition], limit: int = 10) -> list[dict[str, Any]]:
        points, _ = self.client.scroll(
            self.docs_collection, scroll_filter=models.Filter(must=conditions), limit=limit, with_payload=True
        )
        return [dict(p.payload or {}) for p in points]

    def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        found = self._find_docs([models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash))], 1)
        return found[0] if found else None

    def find_documents_by_lsh(self, band_keys: list[str], limit: int = 20) -> list[dict[str, Any]]:
        return self._find_docs([models.FieldCondition(key="lsh", match=models.MatchAny(any=band_keys))], limit)

    def list_documents(self, corpus: str | None = None, limit: int = 50, offset: str | None = None):
        flt = build_filter(corpus=corpus)
        points, next_offset = self.client.scroll(
            self.docs_collection, scroll_filter=flt, limit=limit, offset=offset, with_payload=True
        )
        return [dict(p.payload or {}) for p in points], (str(next_offset) if next_offset else None)

    def count_documents(self, corpus: str | None = None) -> int:
        return self.client.count(self.docs_collection, count_filter=build_filter(corpus=corpus), exact=True).count
