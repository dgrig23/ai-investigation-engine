"""BM25 keyword scoring, implemented by hand and stored as Qdrant sparse vectors.

BM25 score of document d for query q:

    score(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

We split this formula in two:
- The document side (term-frequency saturation + length normalization) is computed here,
  at indexing time, and stored as the weights of a sparse vector {term_id: weight}.
- The IDF side is applied by Qdrant at query time (sparse vector `modifier=IDF`), because IDF
  depends on the whole collection and must stay correct as documents are added and deleted.
Query vectors carry weight 1.0 per unique term, so the dot product equals the BM25 sum.

Tokenization keeps identifiers like "CHG-1042" or "payment-gateway" as whole tokens *and* adds
their parts ("chg", "1042"), so both exact-ID and partial matches work.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in
    into is it its me my no not of on or our she so than that the their them then there these they this to
    was we were what when where which who whom why will with would you your""".split()
)


def light_stem(token: str) -> str:
    """A tiny, predictable suffix stripper (not a full Porter stemmer).

    It only needs to make 'failures'/'failure', 'deployed'/'deploy', 'timeouts'/'timeout' match.
    Numbers and identifiers are left untouched.
    """
    if len(token) <= 4 or not token.isalpha():
        return token
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "es" and not token.endswith(("ses", "xes", "zes", "ches", "shes")):
                continue  # "services" -> handled by the "s" rule below -> "service"
            if suffix == "s" and token.endswith("ss"):
                return token
            return token[: len(token) - len(suffix)] + replacement
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if re.search(r"[-_.]", raw):
            tokens.append(raw)  # keep the full identifier
            tokens.extend(light_stem(p) for p in re.split(r"[-_.]", raw) if p and p not in STOPWORDS)
        else:
            tokens.append(light_stem(raw))
    return tokens


def term_id(token: str) -> int:
    """Stable 31-bit id for a token (hashing trick: no vocabulary file to store or sync)."""
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 90.0):
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        dl = len(tokens)
        tf = Counter(tokens)
        weights: dict[int, float] = {}
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        for token, freq in tf.items():
            w = freq * (self.k1 + 1) / (freq + norm)
            tid = term_id(token)
            weights[tid] = weights.get(tid, 0.0) + w  # (hash collisions are summed; very rare)
        return SparseVector(indices=list(weights), values=list(weights.values()))

    def encode_query(self, text: str) -> SparseVector:
        ids = sorted({term_id(t) for t in tokenize(text)})
        return SparseVector(indices=ids, values=[1.0] * len(ids))
