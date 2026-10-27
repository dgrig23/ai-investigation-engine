"""Ingestion service: add, update and delete documents.

    file ─ load (format-specific) ─ metadata ─ exact dedup ─ near-dup check ─ chunk ─ embed + BM25 ─ Qdrant

Update strategy ("write new, then delete old"): chunk IDs include the version number, so the new
version's chunks are written *before* the old ones are deleted. Search never sees a document
with zero chunks, and a crash mid-update leaves the old version intact.
Re-uploading identical content is a no-op (content hash unchanged), so ingestion is idempotent.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from app.agent.security import is_suspicious
from app.index.bm25 import BM25Encoder
from app.index.embeddings import Embedder
from app.index.store import VectorStore, build_filter, chunk_point_id, date_to_int
from app.ingestion.chunking import get_chunker
from app.ingestion.dedup import LSHIndex, content_hash, lsh_band_keys, minhash_signature, shingles
from app.ingestion.loaders import load_file
from app.ingestion.metadata import build_metadata, extract_entities
from app.models import Chunk, Document

log = logging.getLogger(__name__)
NEAR_DUP_THRESHOLD = 0.8


@dataclass
class IngestResult:
    doc_id: str
    status: str  # added | updated | unchanged | duplicate
    version: int = 0
    chunks: int = 0
    duplicate_of: str | None = None
    near_duplicates: list[dict[str, Any]] = field(default_factory=list)
    title: str = ""


def make_doc_id(corpus: str, filename: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename).strip("_").lower()
    return f"{corpus}:{slug}"


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class IngestionService:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25Encoder,
                 chunk_strategy: str, max_words: int = 220, overlap_words: int = 40):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.chunker = get_chunker(chunk_strategy, max_words, overlap_words)

    # ------------------------------------------------------------- public API

    def ingest_file(self, filename: str, data: bytes, corpus: str = "uploads",
                    doc_id: str | None = None, protected: bool = False) -> IngestResult:
        loaded = load_file(filename, data)
        meta = build_metadata(filename, loaded.text, loaded.metadata)
        doc = Document(
            doc_id=doc_id or make_doc_id(corpus, filename), title=loaded.title, text=loaded.text,
            corpus=corpus, source=filename, format=meta.get("format", ""), metadata=meta,
        )
        return self.ingest_document(doc, protected=protected)

    def ingest_document(self, doc: Document, protected: bool = False) -> IngestResult:
        doc.content_hash = content_hash(doc.text)
        existing = self.store.get_document(doc.doc_id)
        if existing and existing.get("content_hash") == doc.content_hash:
            return IngestResult(doc.doc_id, "unchanged", existing.get("version", 1), existing.get("chunk_count", 0),
                                title=doc.title)

        twin = self.store.find_document_by_hash(doc.content_hash)
        if twin and twin["doc_id"] != doc.doc_id:
            log.info("exact duplicate %s == %s, not indexed", doc.doc_id, twin["doc_id"])
            return IngestResult(doc.doc_id, "duplicate", duplicate_of=twin["doc_id"], title=doc.title)

        band_keys = lsh_band_keys(minhash_signature(doc.text))
        near = self._near_duplicates(doc, band_keys)

        version = int(existing.get("version", 0)) + 1 if existing else 1
        chunks = self.make_chunks(doc, version)
        self._index_chunks(chunks)
        if existing:  # new chunks are in place; now remove the previous version's chunks
            self.store.client.delete(
                self.store.chunks_collection,
                points_selector=models.FilterSelector(filter=models.Filter(
                    must=build_filter(doc_ids=[doc.doc_id]).must,
                    must_not=[models.FieldCondition(key="version", match=models.MatchValue(value=version))],
                )),
                wait=True,
            )
        self.store.upsert_document(self._doc_record(doc, version, len(chunks), band_keys, near, protected, existing))
        return IngestResult(doc.doc_id, "updated" if existing else "added", version, len(chunks),
                            near_duplicates=near, title=doc.title)

    def delete(self, doc_id: str) -> bool:
        if not self.store.get_document(doc_id):
            return False
        self.store.delete_doc_chunks(doc_id)
        self.store.delete_document(doc_id)
        return True

    def ingest_bulk(self, docs: list[Document], batch_size: int = 512, protected: bool = True,
                    detect_near_duplicates: bool = True) -> dict[str, int]:
        """Fast path for large datasets: in-memory dedup, batched embedding, batched upserts."""
        stats = {"input": len(docs), "indexed": 0, "exact_duplicates": 0, "near_duplicates": 0, "chunks": 0}
        seen_hashes: set[str] = set()
        seen_ids: set[str] = set()
        lsh = LSHIndex(threshold=NEAR_DUP_THRESHOLD)
        unique: list[tuple[Document, list[str], list[dict[str, Any]]]] = []
        for doc in docs:
            doc.content_hash = content_hash(doc.text)
            if doc.content_hash in seen_hashes or doc.doc_id in seen_ids:
                stats["exact_duplicates"] += 1
                continue
            seen_hashes.add(doc.content_hash)
            seen_ids.add(doc.doc_id)
            sig = minhash_signature(doc.text)
            near = []
            if detect_near_duplicates:
                near = [{"doc_id": d, "similarity": round(s, 3)} for d, s in lsh.query(sig)]
                stats["near_duplicates"] += bool(near)
                lsh.add(doc.doc_id, sig)
            unique.append((doc, lsh_band_keys(sig), near))

        started = time.time()
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            chunks: list[Chunk] = []
            records = []
            for doc, bands, near in batch:
                doc_chunks = self.make_chunks(doc, version=1)
                chunks.extend(doc_chunks)
                records.append(self._doc_record(doc, 1, len(doc_chunks), bands, near, protected, None))
            self._index_chunks(chunks)
            self.store.upsert_documents(records)
            stats["indexed"] += len(batch)
            stats["chunks"] += len(chunks)
            rate = stats["indexed"] / max(time.time() - started, 1e-6)
            log.info("bulk ingest: %d/%d docs (%.0f docs/s)", stats["indexed"], len(unique), rate)
        return stats

    # ------------------------------------------------------------- internals

    def make_chunks(self, doc: Document, version: int) -> list[Chunk]:
        spans = self.chunker.split(doc.text, record_based=bool(doc.metadata.get("record_based")))
        title_entities = extract_entities(doc.title)
        base = {
            "doc_type": doc.metadata.get("doc_type", "document"),
            "date": doc.metadata.get("date"),
            "date_int": date_to_int(doc.metadata.get("date")),
            "source": doc.source,
            "format": doc.format,
            "version": version,
        }
        chunks = []
        for span in spans:
            entities = list(dict.fromkeys(title_entities + extract_entities(span.text)))
            meta = {**base, "entities": entities, "suspicious": is_suspicious(span.text)}
            chunks.append(Chunk(
                chunk_id=chunk_point_id(doc.doc_id, version, span.position), doc_id=doc.doc_id, corpus=doc.corpus,
                text=span.text, position=span.position, section=span.section, title=doc.title, metadata=meta,
            ))
        return chunks

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.context_text() for c in chunks]
        dense = self.embedder.embed_passages(texts)
        sparse = [self.bm25.encode_document(t) for t in texts]
        self.store.upsert_chunks(chunks, dense, sparse)

    def _near_duplicates(self, doc: Document, band_keys: list[str]) -> list[dict[str, Any]]:
        """LSH gives candidates; we confirm each with the exact Jaccard similarity."""
        near = []
        for cand in self.store.find_documents_by_lsh(band_keys):
            if cand["doc_id"] == doc.doc_id:
                continue
            cand_text = "\n".join(c.text for c in self.store.get_doc_chunks(cand["doc_id"]))
            sim = jaccard(doc.text, cand_text)
            if sim >= NEAR_DUP_THRESHOLD:
                near.append({"doc_id": cand["doc_id"], "similarity": round(sim, 3)})
        return near

    @staticmethod
    def _doc_record(doc: Document, version: int, n_chunks: int, band_keys: list[str],
                    near: list[dict[str, Any]], protected: bool, existing: dict[str, Any] | None) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        meta = {k: v for k, v in doc.metadata.items() if k not in {"entities", "dates"}}
        return {
            "doc_id": doc.doc_id, "corpus": doc.corpus, "title": doc.title, "source": doc.source,
            "format": doc.format, "version": version, "content_hash": doc.content_hash, "lsh": band_keys,
            "chunk_count": n_chunks, "near_duplicates": near, "protected": protected,
            "metadata": meta, "entities": doc.metadata.get("entities", [])[:25],
            "created_at": (existing or {}).get("created_at", now), "updated_at": now,
        }
