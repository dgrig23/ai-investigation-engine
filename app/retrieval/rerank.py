"""Cross-encoder re-ranking.

Bi-encoders (our embedding model) embed query and passage *separately* - fast, but coarse.
A cross-encoder reads the (query, passage) pair *together* with full attention, which is far more
accurate but too slow to run over 70k passages. So we use it only on the top ~30 fused candidates.
Model: ms-marco-MiniLM-L-6-v2 (ONNX, CPU, ~25 ms per batch of 30 on a laptop).
Its raw score is a logit; we also use the best score as a signal for "is there any good evidence?".
"""

from __future__ import annotations

import threading

from app.models import SearchHit


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_chars: int = 800):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)
        # Cost grows with passage length (attention is quadratic and a batch is padded to its longest
        # passage). The first ~800 characters carry the gist; measured: 30 passages ≈ 0.3 s on CPU.
        self.max_chars = max_chars
        self._lock = threading.Lock()  # ONNX sessions are shared; keep calls serialized

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        with self._lock:
            scores = list(self._model.rerank(query, [h.chunk.context_text()[: self.max_chars] for h in hits]))
        for hit, score in zip(hits, scores, strict=True):
            hit.scores["rerank"] = float(score)
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)
