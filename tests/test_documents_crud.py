"""Add / update / delete / duplicate handling against a real (in-memory) Qdrant."""

TEXT_V1 = b"# Runbook\n\nThe default pool size is 200. Restart pods if timeouts rise. " + b"Extra context words. " * 30
TEXT_V2 = b"# Runbook\n\nThe default pool size is 120. Restart pods if timeouts rise. " + b"Different context now. " * 30


def test_add_then_unchanged(ingestion, store):
    first = ingestion.ingest_file("rb.md", TEXT_V1, corpus="t")
    assert first.status == "added" and first.version == 1 and first.chunks >= 1
    again = ingestion.ingest_file("rb.md", TEXT_V1, corpus="t")
    assert again.status == "unchanged"
    assert store.count_documents("t") == 1


def test_update_replaces_chunks(ingestion, store):
    ingestion.ingest_file("rb.md", TEXT_V1, corpus="t")
    res = ingestion.ingest_file("rb.md", TEXT_V2, corpus="t")
    assert res.status == "updated" and res.version == 2
    chunks = store.get_doc_chunks("t:rb.md")
    assert chunks and all(c.metadata["version"] == 2 for c in chunks)
    assert all("200" not in c.text for c in chunks)


def test_exact_duplicate_under_other_name_is_not_indexed(ingestion, store):
    ingestion.ingest_file("rb.md", TEXT_V1, corpus="t")
    dup = ingestion.ingest_file("rb (1).md", TEXT_V1, corpus="t")
    assert dup.status == "duplicate" and dup.duplicate_of == "t:rb.md"
    assert store.count_documents("t") == 1


def test_near_duplicate_is_kept_and_linked(ingestion):
    body = " ".join(f"token{i}" for i in range(400))
    ingestion.ingest_file("a.md", f"# Report\n\n{body} final".encode(), corpus="t")
    res = ingestion.ingest_file("b.md", f"# Report\n\n{body} draft".encode(), corpus="t")
    assert res.status == "added"
    assert res.near_duplicates and res.near_duplicates[0]["doc_id"] == "t:a.md"


def test_delete(ingestion, store):
    ingestion.ingest_file("rb.md", TEXT_V1, corpus="t")
    assert ingestion.delete("t:rb.md") is True
    assert store.get_document("t:rb.md") is None
    assert store.get_doc_chunks("t:rb.md") == []
    assert ingestion.delete("t:rb.md") is False
