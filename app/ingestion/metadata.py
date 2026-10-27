"""Rule-based metadata extraction (no LLM cost).

Extracts the fields that power filtering and cross-document linking:
- entity IDs such as INC-2041 or CHG-1042 and service names such as payment-gateway,
- dates (ISO and written forms), normalized to YYYY-MM-DD,
- a document type inferred from metadata or the file name.
Entities are stored on every chunk, so the agent can jump from one document to every
other document that mentions the same change, incident or service.
"""

from __future__ import annotations

import re
from typing import Any

ID_RE = re.compile(r"\b[A-Z]{2,6}-\d{2,6}\b")
SERVICE_RE = re.compile(
    r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*-(?:service|gateway|engine|api|aggregator|scoring|cluster|queue|cache|worker|feed|db)\b"
)
ISO_DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
WRITTEN_DATE_RE = re.compile(
    r"\b(?:(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(20\d{2})|(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(20\d{2}))\b",
    re.I,
)

DOC_TYPE_KEYWORDS = [
    ("postmortem", "postmortem"),
    ("post-mortem", "postmortem"),
    ("incident", "incident_report"),
    ("change", "change_record"),
    ("runbook", "runbook"),
    ("architecture", "architecture"),
    ("status", "status_update"),
    ("slack", "chat_export"),
    ("chat", "chat_export"),
    ("weekly", "ops_report"),
    ("report", "report"),
    ("email", "email"),
    ("catalog", "service_catalog"),
    ("notes", "notes"),
]


def extract_entities(text: str, limit: int = 40) -> list[str]:
    found: dict[str, None] = {}
    for m in ID_RE.findall(text):
        found.setdefault(m.upper(), None)
    for m in SERVICE_RE.findall(text):
        found.setdefault(m.lower(), None)
    return list(found)[:limit]


def extract_dates(text: str, limit: int = 20) -> list[str]:
    dates: set[str] = set()
    for y, m, d in ISO_DATE_RE.findall(text):
        dates.add(f"{y}-{m}-{d}")
    for d1, mon1, y1, mon2, d2, y2 in WRITTEN_DATE_RE.findall(text):
        day, mon, year = (d1, mon1, y1) if d1 else (d2, mon2, y2)
        dates.add(f"{year}-{MONTHS[mon.lower()]:02d}-{int(day):02d}")
    return sorted(dates)[:limit]


def infer_doc_type(filename: str, metadata: dict[str, Any]) -> str:
    if metadata.get("doc_type"):
        return str(metadata["doc_type"]).strip().lower().replace(" ", "_")
    name = filename.lower()
    for keyword, doc_type in DOC_TYPE_KEYWORDS:
        if keyword in name:
            return doc_type
    return "document"


def build_metadata(filename: str, text: str, file_meta: dict[str, Any]) -> dict[str, Any]:
    """Merge format-specific metadata with rule-based extraction."""
    meta = {k: v for k, v in file_meta.items() if isinstance(v, str | int | float | bool)}
    meta["doc_type"] = infer_doc_type(filename, file_meta)
    meta["entities"] = extract_entities(text)
    dates = extract_dates(text)
    meta["dates"] = dates
    if not meta.get("date") and dates:
        meta["date"] = dates[0]
    return meta
