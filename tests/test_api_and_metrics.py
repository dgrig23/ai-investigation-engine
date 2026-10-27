import pytest
from fastapi.testclient import TestClient

from app.agent.investigator import AgentConfig, Investigator
from app.index.bm25 import BM25Encoder
from app.retrieval.retriever import RetrievalConfig
from app.services import Services
from eval import metrics as m
from tests.conftest import FakeLLM

# ------------------------------------------------------------------ metrics


def test_retrieval_metrics():
    retrieved = ["a", "x", "b", "y"]
    assert m.recall_at_k(retrieved, ["a", "b"], 1) == 0.5
    assert m.recall_at_k(retrieved, ["a", "b"], 3) == 1.0
    assert m.all_recall_at_k(retrieved, ["a", "b"], 2) == 0.0
    assert m.all_recall_at_k(retrieved, ["a", "b"], 3) == 1.0
    assert m.mrr(retrieved, ["b"]) == pytest.approx(1 / 3)
    assert m.mrr(retrieved, ["z"]) == 0.0
    assert m.ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_answer_metrics():
    assert m.exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert m.token_f1("Paris France", "Paris") == pytest.approx(2 / 3)
    assert m.key_fact_recall("It lasted 107 minutes, CHG-1042", ["107 minutes|107 min", "CHG-1051"]) == 0.5
    assert m.fact_present("41,300 deposits", "41300")
    assert m.citation_precision(["a", "b"], ["a"]) == 0.5
    assert m.citation_precision([], ["a"]) is None
    assert m.citation_recall(["a"], ["a", "b"]) == 0.5


# ------------------------------------------------------------------ API


@pytest.fixture
def client(settings, store, embedder, ingestion, retriever, monkeypatch):
    import app.api.main as api

    svc = Services(settings)
    llm = FakeLLM([[("submit_answer", {"answer": "The pool size is 200 [S1].", "status": "answered",
                                       "claims": [{"statement": "The pool size is 200.", "citations": ["S1"]}]})]] * 5)
    llm.describe = lambda: [{"provider": "fake", "model": "fake"}]
    svc.__dict__.update(store=store, embedder=embedder, reranker=None, bm25=BM25Encoder(), ingestion=ingestion,
                        retriever=retriever, llm=llm, retrieval_config=RetrievalConfig(use_reranker=False))
    svc.__dict__["investigator"] = Investigator(retriever, store, llm, None, AgentConfig(retrieval=RetrievalConfig(use_reranker=False)))
    svc.__dict__["agent_config"] = AgentConfig(retrieval=RetrievalConfig(use_reranker=False))
    monkeypatch.setattr(api, "get_services", lambda: svc)
    api.app.dependency_overrides[api.services] = lambda: svc
    api.answer_cache.clear()
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["llm_providers"][0]["provider"] == "fake"


def test_upload_list_update_delete(client):
    r = client.post("/api/documents", files=[("files", ("rb.md", b"# Runbook\n\nThe default pool size is 200.", "text/markdown"))],
                    data={"corpus": "uploads"})
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "added"
    docs = client.get("/api/documents", params={"corpus": "uploads"}).json()["documents"]
    assert [d["doc_id"] for d in docs] == ["uploads:rb.md"]
    r = client.put("/api/documents/uploads:rb.md", files={"file": ("rb.md", b"# Runbook\n\nThe default pool size is 120.", "text/markdown")})
    assert r.json()["status"] == "updated" and r.json()["version"] == 2
    detail = client.get("/api/documents/uploads:rb.md").json()
    assert "120" in detail["chunks"][0]["text"]
    assert client.delete("/api/documents/uploads:rb.md").status_code == 200
    assert client.get("/api/documents/uploads:rb.md").status_code == 404


def test_upload_rejects_unsupported_format(client):
    r = client.post("/api/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.json()["results"][0]["status"] == "error"


def test_protected_corpus_is_read_only(client):
    r = client.post("/api/documents", files=[("files", ("a.md", b"# A\n\ntext", "text/markdown"))], data={"corpus": "incidents"})
    assert r.status_code == 403


def test_search_and_ask(client):
    client.post("/api/documents", files=[("files", ("rb.md", b"# Runbook\n\nThe default pool size is 200.", "text/markdown"))])
    hits = client.get("/api/search", params={"q": "pool size", "corpus": "uploads", "rerank": False}).json()["hits"]
    assert hits and hits[0]["doc_id"] == "uploads:rb.md"
    r = client.post("/api/ask", json={"question": "What is the default pool size?", "corpus": "uploads"})
    body = r.json()
    assert body["status"] == "answered" and body["sources"][0]["doc_id"] == "uploads:rb.md"
    again = client.post("/api/ask", json={"question": "What is the default pool size?", "corpus": "uploads"}).json()
    assert again.get("cached") is True


def test_ask_stream_emits_events(client):
    client.post("/api/documents", files=[("files", ("rb.md", b"# Runbook\n\nThe default pool size is 200.", "text/markdown"))])
    with client.stream("POST", "/api/ask/stream", json={"question": "pool size?", "corpus": "uploads"}) as r:
        text = "".join(r.iter_text())
    assert '"type": "tool_call"' in text and '"type": "final"' in text


def test_validation_errors(client):
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.get("/api/search", params={"q": "pool", "mode": "nope"}).status_code == 422
