"""Duplicate detection, written from scratch.

Two levels:
1. Exact duplicates: SHA-256 of the normalized text. The same report uploaded twice (or the
   same Wikipedia paragraph appearing in many HotpotQA questions) is stored once.
2. Near-duplicates: MinHash + Locality-Sensitive Hashing (LSH) over word 5-gram "shingles".
   - Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| of the two shingle sets.
   - MinHash: apply P random hash functions to every shingle and keep the minimum of each.
     For one hash function, P[min(A) == min(B)] = J(A, B), so the fraction of equal signature
     positions estimates the Jaccard similarity.
   - LSH: cut the signature into B bands of R rows. Two documents become candidates if any band
     is identical. P(candidate) = 1 - (1 - J^R)^B, an S-curve with threshold ≈ (1/B)^(1/R).
     With 128 permutations as 16 bands × 8 rows the threshold is ≈ 0.71.
   Near-duplicates are *kept* but linked: in an investigation, a revised report that differs in
   one number is exactly the kind of signal we want to surface, not silently drop.
"""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

NUM_PERM = 128
BANDS = 16
ROWS = NUM_PERM // BANDS
SHINGLE_SIZE = 5
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1

# Fixed seed -> the same document always gets the same signature (reproducible across runs).
_rng = np.random.RandomState(1337)
_A = _rng.randint(1, _MERSENNE_PRIME, size=NUM_PERM, dtype=np.uint64)
_B = _rng.randint(0, _MERSENNE_PRIME, size=NUM_PERM, dtype=np.uint64)

_WORD_RE = re.compile(r"[a-z0-9]+")


def content_hash(text: str) -> str:
    canonical = " ".join(_WORD_RE.findall(text.lower()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(text: str) -> np.ndarray:
    """128 MinHash values using universal hashing h(x) = (a*x + b) mod p, vectorized with numpy."""
    sh = shingles(text)
    if not sh:
        return np.full(NUM_PERM, _MAX_HASH, dtype=np.uint64)
    hashes = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
    # (NUM_PERM, 1) * (1, n) -> (NUM_PERM, n); uint64 arithmetic wraps, which is fine for hashing.
    permuted = (_A[:, None] * hashes[None, :] + _B[:, None]) % np.uint64(_MERSENNE_PRIME)
    return (permuted & np.uint64(_MAX_HASH)).min(axis=1)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def lsh_band_keys(signature: np.ndarray) -> list[str]:
    """One string key per band; identical keys mean 'candidate near-duplicate'."""
    keys = []
    for band in range(BANDS):
        chunk = signature[band * ROWS:(band + 1) * ROWS].tobytes()
        keys.append(f"{band}:{hashlib.blake2b(chunk, digest_size=8).hexdigest()}")
    return keys


class LSHIndex:
    """In-memory LSH index used for bulk ingestion (the API path queries the vector store instead)."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._buckets: dict[str, list[str]] = {}
        self._signatures: dict[str, np.ndarray] = {}

    def add(self, doc_id: str, signature: np.ndarray) -> None:
        self._signatures[doc_id] = signature
        for key in lsh_band_keys(signature):
            self._buckets.setdefault(key, []).append(doc_id)

    def query(self, signature: np.ndarray) -> list[tuple[str, float]]:
        candidates: set[str] = set()
        for key in lsh_band_keys(signature):
            candidates.update(self._buckets.get(key, []))
        scored = [(c, estimated_jaccard(signature, self._signatures[c])) for c in candidates]
        return sorted([s for s in scored if s[1] >= self.threshold], key=lambda x: -x[1])
