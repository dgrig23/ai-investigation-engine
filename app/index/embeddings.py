"""Dense embeddings with a small local model (BAAI/bge-small-en-v1.5, 384 dims, ONNX on CPU).

Why local: it is free, has no rate limits (we embed ~70k paragraphs), and the exact same model
runs at query time in the deployed container. BGE models are trained with an instruction
prefix for *queries* (not for passages), which measurably improves retrieval.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, cache_size: int = 2048, threads: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def embed_passages(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        """Embed passages, batching texts of similar length together.

        A batch is padded to its longest text, so one 400-token paragraph makes 31 short neighbours
        cost 400 tokens each. Sorting by length first (then restoring the order) made bulk
        embedding of HotpotQA several times faster on CPU.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = list(self._model.embed([texts[i] for i in order], batch_size=batch_size))
        out = np.empty((len(texts), len(vectors[0])), dtype=np.float32)
        for pos, i in enumerate(order):
            out[i] = vectors[pos]
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Query embeddings are cached (LRU): repeated questions and agent sub-queries are free."""
        return self._embed_query_cached(text)

    def _embed_query_cached(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        vector = np.asarray(next(iter(self._model.embed([self._query_prefix + text]))), dtype=np.float32)
        with self._lock:
            self._cache[text] = vector
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vector


class CachedPassageEmbedder:
    """Disk cache for passage embeddings (keyed by a hash of the exact text).

    Embedding 66k paragraphs on a laptop CPU takes minutes; with the cache, re-indexing into another
    Qdrant (local server, Qdrant Cloud) or with other settings costs no embedding time at all.
    """

    def __init__(self, embedder: Embedder, path: str):
        self.embedder = embedder
        self.model_name = embedder.model_name
        self.path = path
        self._vectors: dict[str, np.ndarray] = {}
        try:
            data = np.load(path, allow_pickle=False)
            self._vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        except (FileNotFoundError, OSError, KeyError):
            pass

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embed_passages(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        if missing:
            new = self.embedder.embed_passages([texts[i] for i in missing], batch_size=batch_size)
            for i, vec in zip(missing, new, strict=True):
                self._vectors[keys[i]] = vec
        return np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embedder.embed_query(text)

    def save(self) -> None:
        keys = np.array(list(self._vectors), dtype="U40")
        vectors = np.stack(list(self._vectors.values())).astype(np.float32)
        np.savez(self.path, keys=keys, vectors=vectors)

    def __len__(self) -> int:
        return len(self._vectors)
