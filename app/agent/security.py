"""Prompt-injection detection for retrieved content (indirect prompt injection).

In RAG, the dangerous text is usually not the user's question but a *document* that says
"ignore previous instructions...". Our defence has three layers:
1. Detection (here): pattern-based scoring flags suspicious chunks at ingestion and at query time.
2. Isolation: retrieved text is wrapped in <document> tags and the system prompt says that
   document content is untrusted data, never instructions.
3. Neutralization: flagged chunks are shown to the model with an explicit warning label, and
   role-like markers (e.g. "system:", "<|im_start|>") are defanged.
The evaluation contains injection questions and measures the attack success rate.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|rules|guidelines)",
    r"forget (all |your )?(previous |prior )?instructions",
    r"you are now (a|an|the) ",
    r"new instructions?:",
    r"(reveal|print|show|output) (your|the) (system )?prompt",
    r"(ai|llm|assistant|chatbot|language model)s?\b.{0,40}\b(must|should|always) (say|answer|respond|state|reply)",
    r"when (asked|answering).{0,60}(always|you must) (say|answer|respond|state)",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"^\s*(system|assistant)\s*:",
    r"do not (cite|mention) (any )?(other )?(sources|documents)",
]
_COMPILED = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]
_ROLE_MARKERS = re.compile(r"<\|?(im_start|im_end|system|endoftext)\|?>|^\s*(system|assistant)\s*:", re.I | re.M)


def detect_injection(text: str) -> list[str]:
    """Return the patterns that matched (empty list = looks clean)."""
    return [p.pattern for p in _COMPILED if p.search(text)]


def is_suspicious(text: str) -> bool:
    return bool(detect_injection(text))


def neutralize(text: str) -> str:
    """Defang role markers so they cannot be confused with real conversation turns."""
    return _ROLE_MARKERS.sub(lambda m: "[removed-role-marker]", text)
