"""Maximal Marginal Relevance (MMR) - our own implementation.

Multi-document questions fail when the top-k is five near-identical chunks from one document.
MMR picks results one at a time, trading relevance against redundancy:

    next = argmax_{d ∉ S}  λ · rel(d) − (1 − λ) · max_{s ∈ S} cos(d, s)

λ = 1 gives plain relevance order; lower λ gives more diverse results.
"""

from __future__ import annotations

import numpy as np

from app.models import SearchHit


def mmr_select(hits: list[SearchHit], k: int, lam: float = 0.75) -> list[SearchHit]:
    if len(hits) <= 1 or k <= 0:
        return hits[:k]
    usable = [h for h in hits if h.vector is not None]
    if len(usable) < len(hits):  # without vectors we cannot measure redundancy; keep relevance order
        return hits[:k]

    rel = np.array([h.score for h in hits], dtype=np.float64)
    rel = (rel - rel.min()) / (rel.max() - rel.min()) if rel.max() > rel.min() else np.ones_like(rel)
    vecs = np.array([h.vector for h in hits], dtype=np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    sim = vecs @ vecs.T

    selected: list[int] = [int(np.argmax(rel))]
    remaining = set(range(len(hits))) - set(selected)
    while remaining and len(selected) < k:
        best, best_score = None, -np.inf
        for i in remaining:
            score = lam * rel[i] - (1 - lam) * max(sim[i, j] for j in selected)
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.remove(best)
    for rank, i in enumerate(selected, start=1):
        hits[i].scores["mmr_rank"] = float(rank)
    return [hits[i] for i in selected]
