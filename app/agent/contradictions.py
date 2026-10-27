"""Rule-based contradiction detection over retrieved evidence (no LLM cost).

Idea: extract simple (entity, attribute, value) facts from each chunk - e.g. the duration or
start time of incident INC-2041, or its stated root cause - and compare facts about the same
entity and attribute that come from *different documents*. Disagreements become "conflict hints"
that are shown to the model and merged into the final answer if the model missed them.

The LLM also looks for contradictions itself; this detector is a cheap, deterministic safety net
that catches the most common investigation conflicts: numbers, times and causes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from app.agent.evidence import Evidence

DURATION_RE = re.compile(
    r"(?:lasted|duration(?:\s+of)?|for a total of|total(?:ling)?|outage of|impact of|for)\s*(?:of\s*)?(?:approximately|about|roughly|around|~)?\s*"
    r"(\d+(?:\.\d+)?)\s*(minutes|mins|min|hours|hrs|hour|h)\b",
    re.I,
)
START_RE = re.compile(
    r"(?:started|began|start(?:ed)? at|starting at|first (?:alerts?|errors?)(?: fired)?(?: at)?|impact began(?: at)?|"
    r"onset)\D{0,25}?(\d{1,2}:\d{2})\s*UTC",
    re.I,
)
ROOT_CAUSE_RE = re.compile(r"root cause\s*(?:was|is|:|\u2014|-)\s*([^.\n]{12,180})", re.I)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "by", "was", "is", "that", "for", "with", "from"}


@dataclass
class Fact:
    entity: str
    attribute: str
    value: str
    sid: str
    doc_id: str


@dataclass
class ConflictHint:
    entity: str
    attribute: str
    values: list[tuple[str, str]]  # (value, evidence id)
    most_reliable: tuple[str, str, str] | None = None  # (value, evidence id, why)

    def describe(self) -> str:
        grouped: dict[str, list[str]] = {}
        for value, sid in self.values:  # "107 min [S1, S3]; 45 min [S5]"
            grouped.setdefault(value, []).append(sid)
        vals = "; ".join(f"{v} [{', '.join(sids)}]" for v, sids in grouped.items())
        text = f"{self.entity} {self.attribute}: {vals}"
        if self.most_reliable:
            value, sid, why = self.most_reliable
            text += f" - most reliable: {value} [{sid}] ({why})"
        return text


# How authoritative a document type is for facts about an incident (higher = more reliable).
# A reviewed postmortem is written after the investigation; an initial report or a chat message is
# written while facts are still unknown.
DOC_TYPE_RELIABILITY = {
    "postmortem": 5.0, "report": 4.5, "change_record": 4.0, "alerts": 4.0, "runbook": 3.5,
    "incident_report": 3.0, "service_catalog": 2.5, "architecture": 2.5, "ops_report": 2.0,
    "status_update": 2.0, "notes": 2.0, "chat_export": 1.5, "email": 1.0,
}


def source_reliability(ev: Evidence) -> tuple[float, str]:
    """Score a source: document type, minus a penalty for drafts/preliminary numbers, plus recency."""
    meta = ev.chunk.metadata
    doc_type = str(meta.get("doc_type") or "document")
    score = DOC_TYPE_RELIABILITY.get(doc_type, 2.0)
    reasons = [doc_type.replace("_", " ")]
    label = f"{ev.chunk.title} {meta.get('status', '')}".lower()
    opening = ev.chunk.text[:200].lower()
    if any(w in label for w in ("draft", "initial", "preliminary")) or any(
            w in opening for w in ("draft", "preliminary", "not yet confirmed")):
        score -= 1.5
        reasons.append("draft/preliminary")
    date_int = meta.get("date_int") or 0
    if date_int:
        score += (date_int % 10000) / 100000  # tiny tie-breaker: newer wins between equal types
        reasons.append(str(meta.get("date")))
    return score, ", ".join(reasons)


def _minutes(number: str, unit: str) -> float:
    n = float(number)
    return n * 60 if unit.lower().startswith("h") else n


INCIDENT_ID_RE = re.compile(r"\b(?:INC|PM)-\d{3,6}\b")


def _entity_at(text: str, pos: int, default: str | None) -> str | None:
    """The incident a fact belongs to: the closest incident ID *before* it on the same line,
    otherwise the document's main incident (from its title)."""
    line_start = text.rfind("\n", 0, pos) + 1
    ids = INCIDENT_ID_RE.findall(text[line_start:pos])
    if ids:
        return ids[-1].replace("PM-", "INC-")  # postmortem PM-2041 describes incident INC-2041
    return default


def extract_facts(ev: Evidence) -> list[Fact]:
    text = ev.chunk.text
    title_ids = INCIDENT_ID_RE.findall(ev.chunk.title)
    default = title_ids[0].replace("PM-", "INC-") if title_ids else None
    facts = []
    for m in DURATION_RE.finditer(text):
        if entity := _entity_at(text, m.start(), default):
            facts.append(Fact(entity, "duration", f"{_minutes(m.group(1), m.group(2)):.0f} min", ev.sid, ev.chunk.doc_id))
    for m in START_RE.finditer(text):
        if entity := _entity_at(text, m.start(), default):
            h, mnt = m.group(1).split(":")
            facts.append(Fact(entity, "start time", f"{int(h):02d}:{mnt} UTC", ev.sid, ev.chunk.doc_id))
    for m in ROOT_CAUSE_RE.finditer(text):
        if entity := _entity_at(text, m.start(), default):
            facts.append(Fact(entity, "root cause", m.group(1).strip(), ev.sid, ev.chunk.doc_id))
    return facts


def _similar_cause(a: str, b: str) -> bool:
    wa = {w for w in _WORD.findall(a.lower()) if w not in _STOP}
    wb = {w for w in _WORD.findall(b.lower()) if w not in _STOP}
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.25


def find_conflicts(evidence: list[Evidence]) -> list[ConflictHint]:
    by_sid = {ev.sid: ev for ev in evidence}
    facts = [f for ev in evidence for f in extract_facts(ev)]
    groups: dict[tuple[str, str], list[Fact]] = {}
    for f in facts:
        groups.setdefault((f.entity, f.attribute), []).append(f)

    hints = []
    for (entity, attribute), group in groups.items():
        conflicting: dict[str, Fact] = {}
        for a, b in combinations(group, 2):
            if a.doc_id == b.doc_id:
                continue  # one document restating itself is not a contradiction
            same = _similar_cause(a.value, b.value) if attribute == "root cause" else a.value == b.value
            if not same:
                conflicting.setdefault(a.sid + a.value, a)
                conflicting.setdefault(b.sid + b.value, b)
        if conflicting:
            ranked = sorted(conflicting.values(), key=lambda f: -source_reliability(by_sid[f.sid])[0])
            best = ranked[0]
            why = source_reliability(by_sid[best.sid])[1]
            hints.append(ConflictHint(entity, attribute, [(f.value, f.sid) for f in conflicting.values()],
                                      most_reliable=(best.value, best.sid, why)))
    return hints
