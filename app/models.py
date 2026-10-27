"""Core data structures shared by every layer (ingestion, index, retrieval, agent, API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A parsed source document, before chunking."""

    doc_id: str
    title: str
    text: str
    corpus: str
    source: str  # file name or URI it came from
    format: str  # pdf, docx, md, html, txt, csv, json, jsonl
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass
class Chunk:
    """A retrievable piece of a document."""

    chunk_id: str  # deterministic UUID (doc_id + version + position)
    doc_id: str
    corpus: str
    text: str
    position: int
    section: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def context_text(self) -> str:
        """Text used for embedding and BM25: a short header gives the chunk its context."""
        header = self.title if not self.section or self.section == self.title else f"{self.title} > {self.section}"
        return f"{header}\n{self.text}" if header else self.text


@dataclass
class SearchHit:
    """One retrieved chunk plus the scores from every retrieval stage (for transparency)."""

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    vector: list[float] | None = None
