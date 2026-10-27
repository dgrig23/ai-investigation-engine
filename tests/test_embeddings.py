"""Embedder logic without downloading a model: a fake model stands in for the ONNX session."""

import numpy as np

from app.index.embeddings import BGE_QUERY_PREFIX, CachedPassageEmbedder, Embedder


class FakeModel:
    def __init__(self):
        self.seen_batches = []

    def embed(self, texts, batch_size=64):
        self.seen_batches.append(list(texts))
        for t in texts:
            yield np.array([len(t), 1.0], dtype=np.float32)


def make_embedder() -> Embedder:
    e = Embedder.__new__(Embedder)  # skip __init__ (it would load the real model)
    import threading
    from collections import OrderedDict

    e.model_name, e._model, e._query_prefix = "fake-bge", FakeModel(), BGE_QUERY_PREFIX
    e._cache, e._cache_size, e._lock = OrderedDict(), 10, threading.Lock()
    return e


def test_passages_are_sorted_by_length_but_returned_in_input_order():
    e = make_embedder()
    texts = ["ccc", "a", "bbbbb", "dd"]
    out = e.embed_passages(texts)
    assert e._model.seen_batches[0] == ["a", "dd", "ccc", "bbbbb"]  # model saw them sorted
    assert out[:, 0].tolist() == [3, 1, 5, 2]  # caller gets them back in the original order


def test_query_prefix_and_lru_cache():
    e = make_embedder()
    e.embed_query("why")
    e.embed_query("why")
    assert len(e._model.seen_batches) == 1  # second call served from cache
    assert e._model.seen_batches[0] == [BGE_QUERY_PREFIX + "why"]


def test_disk_cache_roundtrip(tmp_path):
    path = str(tmp_path / "emb.npz")
    cache = CachedPassageEmbedder(make_embedder(), path)
    first = cache.embed_passages(["alpha", "be"])
    cache.save()
    reloaded = CachedPassageEmbedder(make_embedder(), path)
    assert len(reloaded) == 2
    again = reloaded.embed_passages(["be", "alpha"])
    assert reloaded.embedder._model.seen_batches == []  # nothing recomputed
    assert np.allclose(again, first[::-1])
