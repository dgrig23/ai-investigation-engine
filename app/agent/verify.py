"""Post-generation verification: never trust the model's citations blindly.

For each claim the model makes we check, without another LLM call:
1. Citation validity - cited ids must exist in the evidence registry (invented ids are removed).
2. Exact-token grounding - every number, time and ID in the claim (e.g. "107 minutes",
   "14:05", "CHG-1042") must literally appear in at least one cited chunk. This catches the most
   damaging hallucinations: wrong numbers attributed to a real source.
3. Lexical + semantic support - content-word overlap and embedding similarity between the claim
   and its cited chunks.
Then the answer-level gate: if nothing is supported, or retrieval confidence is very low and
support is weak, the system abstains ("insufficient evidence") instead of answering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.agent.contradictions import ConflictHint
from app.agent.evidence import EvidenceRegistry
from app.index.bm25 import tokenize

_UNICODE_FIXES = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "\u2013": "-", "\u2014": "-", "−": "-",  # hyphens/dashes
    " ": " ", " ": " ", " ": " ", " ": " ",  # non-breaking / thin spaces
})


def normalize_llm_text(text: str) -> str:
    """Models often write 'CHG‑1042' with a non-breaking hyphen (U+2011), '1 800' with a narrow
    space or '[ S1 ]' with spaces. Normalizing keeps IDs, numbers and citations comparable."""
    text = text.translate(_UNICODE_FIXES)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)  # zero-width characters
    # gpt-oss style "【S8】" / "【S8†L3-L9】" -> "[S8]"
    text = re.sub(r"【\s*(S\d+)[^】]*】", r"[\1]", text)
    # "[ S6 , S13 ]" or "[S6, S13]" -> "[S6][S13]"
    return re.sub(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]",
                  lambda m: "".join(f"[{sid}]" for sid in re.findall(r"S\d+", m.group(1))), text)


EXACT_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}-\d{2,6}\b|\b\d{1,2}:\d{2}\b|\b\d[\d,]*(?:\.\d+)?%?")
CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass
class ClaimCheck:
    statement: str
    citations: list[str]
    support: str  # supported | partial | unsupported
    missing_tokens: list[str] = field(default_factory=list)
    lexical_overlap: float = 0.0
    semantic_similarity: float | None = None
    invalid_citations: list[str] = field(default_factory=list)


def _norm_number(tok: str) -> str:
    return tok.replace(",", "").rstrip("%").rstrip(".")


def _token_in(token: str, text: str) -> bool:
    """Whole-token match: '0' must not match inside '2052'; '1,800' matches '1800'."""
    tok = re.escape(_norm_number(token))
    return re.search(rf"(?<![\w.]){tok}(?![\w])", text.replace(",", "")) is not None


def check_claim(statement: str, citations: list[str], registry: EvidenceRegistry, embedder=None) -> ClaimCheck:
    valid = [c.upper() for c in citations if c in registry]
    invalid = [c for c in citations if c not in registry]
    if not valid:
        return ClaimCheck(statement, [], "unsupported", invalid_citations=invalid)

    cited_texts = [registry.get(c).chunk.context_text() for c in valid]
    joined = " ".join(cited_texts)
    # Ignore numbers that are only part of a citation marker like [S3].
    claim_wo_citations = CITATION_RE.sub("", statement)
    missing = []
    for tok in EXACT_TOKEN_RE.findall(claim_wo_citations):
        if not _token_in(tok, joined):
            missing.append(tok)

    claim_terms = set(tokenize(claim_wo_citations))
    evidence_terms = set(tokenize(joined))
    overlap = len(claim_terms & evidence_terms) / len(claim_terms) if claim_terms else 0.0

    similarity = None
    if embedder is not None:
        vecs = embedder.embed_passages([claim_wo_citations, *cited_texts])
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        similarity = float(np.max(vecs[1:] @ vecs[0]))

    strong = overlap >= 0.5 or (similarity is not None and similarity >= 0.75)
    weak_ok = overlap >= 0.25 or (similarity is not None and similarity >= 0.6)
    # A number, time or ID that appears in none of the cited sources is an unverified fact.
    if missing:
        support = "unsupported"
    elif strong:
        support = "supported"
    elif weak_ok:
        support = "partial"
    else:
        support = "unsupported"
    return ClaimCheck(statement, valid, support, missing, round(overlap, 3),
                      None if similarity is None else round(similarity, 3), invalid)


def attribute_citations(answer: str, registry: EvidenceRegistry) -> tuple[list[ClaimCheck], str]:
    """Match each sentence of an uncited answer to its best-supporting evidence.

    Returns (checks, answer with [S#] appended) when *every* factual sentence is grounded
    (all numbers/IDs found, enough word overlap); otherwise ([], original answer).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    checks: list[ClaimCheck] = []
    cited_answer = answer
    for sentence in sentences:
        if len(sentence.split()) < 4:  # headings, bullets like "Services:" - nothing to verify
            continue
        candidates = [check_claim(sentence, [ev.sid], registry) for ev in registry.all()]
        grounded = [c for c in candidates if not c.missing_tokens and c.support in ("supported", "partial")]
        if not grounded:
            # A synthesis sentence ("A took 107 min, longer than B's 97 min") combines two sources:
            # try pairs of the most overlapping passages.
            top = sorted(candidates, key=lambda c: -c.lexical_overlap)[:6]
            pairs = [check_claim(sentence, [a.citations[0], b.citations[0]], registry)
                     for i, a in enumerate(top) for b in top[i + 1:] if a.citations and b.citations]
            grounded = [c for c in pairs if not c.missing_tokens and c.support in ("supported", "partial")]
        if not grounded:
            return [], answer
        best = max(grounded, key=lambda c: c.lexical_overlap)
        checks.append(best)
        marks = "".join(f"[{sid}]" for sid in best.citations)
        cited_answer = cited_answer.replace(sentence, f"{sentence} {marks}", 1)
    return checks, cited_answer


def verify_answer(
    submitted: dict[str, Any],
    registry: EvidenceRegistry,
    conflict_hints: list[ConflictHint],
    best_rerank_score: float | None,
    abstain_threshold: float,
    embedder=None,
    question: str = "",
) -> dict[str, Any]:
    submitted = json.loads(normalize_llm_text(json.dumps(submitted, ensure_ascii=False)))
    status = str(submitted.get("status", "answered")).lower()
    claims_in = submitted.get("claims") or []
    answer_text = str(submitted.get("answer", "")).strip()

    # If the model gave no structured claims, treat each cited sentence of the answer as a claim.
    if not claims_in and answer_text:
        for sentence in re.split(r"(?<=[.!?])\s+", answer_text):
            cites = CITATION_RE.findall(sentence)
            if cites:
                claims_in.append({"statement": sentence, "citations": cites})

    checks = []
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        cites = [str(c).strip("[] ") for c in claim.get("citations") or []]
        statement = str(claim.get("statement", "")).strip()
        cites += [c for c in CITATION_RE.findall(statement) if c not in cites]
        if statement:
            checks.append(check_claim(statement, cites, registry, embedder))

    supported = [c for c in checks if c.support == "supported"]
    partial = [c for c in checks if c.support == "partial"]
    groundedness = (len(supported) + 0.5 * len(partial)) / len(checks) if checks else 0.0

    warnings = []
    low_retrieval = best_rerank_score is not None and best_rerank_score < abstain_threshold
    if low_retrieval:
        warnings.append(f"Low retrieval confidence (best rerank score {best_rerank_score:.2f}).")
    if status == "answered" and checks and not supported and not partial:
        status = "insufficient_evidence"
        warnings.append("No claim in the draft answer was supported by the cited evidence; abstaining.")
    elif status == "answered" and any(c.support == "unsupported" for c in checks):
        bad = [c.statement[:80] for c in checks if c.support == "unsupported"]
        warnings.append(f"{len(bad)} claim(s) could not be verified against the cited sources: " + " | ".join(bad))
    if status == "answered" and low_retrieval and groundedness < 0.5:
        status = "insufficient_evidence"
        warnings.append("Weak evidence and weakly supported claims; abstaining.")
    elif status == "answered" and not checks:
        # The model answered without citations. Try to attribute every sentence to evidence it has
        # seen (post-hoc attribution). If any sentence cannot be grounded - e.g. a number that appears
        # in no source - the answer is unverifiable and the system abstains instead of guessing.
        auto_checks, answer_text = attribute_citations(answer_text, registry)
        if auto_checks:
            checks = auto_checks
            supported = [c for c in checks if c.support == "supported"]
            partial = [c for c in checks if c.support == "partial"]
            groundedness = (len(supported) + 0.5 * len(partial)) / len(checks)
            warnings.append("Citations were added automatically by matching each sentence to the evidence.")
        else:
            status = "insufficient_evidence"
            warnings.append("The draft answer cited no evidence and could not be matched to the sources; abstaining.")

    contradictions = [c for c in (submitted.get("contradictions") or []) if isinstance(c, dict)]
    covered = " ".join(str(c.get("description", "")) for c in contradictions).lower()
    context = (question + " " + answer_text).upper()
    for hint in conflict_hints:
        if hint.entity.upper() not in context:
            continue  # only report conflicts about something the question or answer is about
        key_terms = [hint.attribute.split()[0], hint.entity.lower()]
        if not all(t.lower() in covered for t in key_terms):
            contradictions.append({
                "description": f"Automatically detected conflict - {hint.describe()}",
                "citations": [sid for _, sid in hint.values],
                "detected_by": "rule-based detector",
            })

    used_ids = sorted({c for ch in checks for c in ch.citations}, key=lambda s: int(s[1:]))
    return {
        "answer": answer_text,
        "status": status,
        "claims": [c.__dict__ for c in checks],
        "contradictions": contradictions,
        "missing_information": submitted.get("missing_information") or "",
        "groundedness": round(groundedness, 3),
        "cited_ids": used_ids,
        "warnings": warnings,
    }
