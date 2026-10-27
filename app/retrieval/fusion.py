"""Weighted Reciprocal Rank Fusion (RRF) - our own implementation.

Dense (semantic) and BM25 (keyword) scores live on different scales (cosine ≈ 0..1, BM25 ≈ 0..30),
so adding them directly is meaningless. RRF uses only the *rank* of each result:

    RRF(d) = Σ_i  w_i / (k + rank_i(d))         rank starts at 1; missing from a list -> no term

k (default 60) damps the advantage of the very top ranks, so a document that is decent in both
lists beats one that is first in one list and absent from the other. Weights let us favour one
retriever (e.g. keywords for queries full of IDs like "CHG-1042").
"""

from __future__ import annotations

from app.models import SearchHit


def weighted_rrf(result_lists: list[list[SearchHit]], weights: list[float] | None = None, k: int = 60) -> list[SearchHit]:
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("one weight per result list is required")

    fused: dict[str, SearchHit] = {}
    totals: dict[str, float] = {}
    for hits, weight in zip(result_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.chunk_id
            totals[cid] = totals.get(cid, 0.0) + weight / (k + rank)
            if cid not in fused:
                fused[cid] = SearchHit(chunk=hit.chunk, score=0.0, scores=dict(hit.scores), vector=hit.vector)
            else:
                fused[cid].scores.update(hit.scores)
                fused[cid].vector = fused[cid].vector or hit.vector
            fused[cid].scores[f"rank_{next(iter(hit.scores), 'list')}"] = float(rank)

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, total in ordered:
        hit = fused[cid]
        hit.score = total
        hit.scores["rrf"] = total
        out.append(hit)
    return out
