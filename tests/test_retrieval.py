from app.index.bm25 import BM25Encoder, light_stem, term_id, tokenize
from app.models import Chunk, SearchHit
from app.retrieval.fusion import weighted_rrf
from app.retrieval.mmr import mmr_select


def _hit(cid: str, score: float = 1.0, vector=None, source: str = "dense") -> SearchHit:
    return SearchHit(chunk=Chunk(chunk_id=cid, doc_id=cid, corpus="t", text=cid, position=0), score=score,
                     scores={source: score}, vector=vector)


def test_tokenize_keeps_identifiers_and_parts():
    toks = tokenize("CHG-1042 changed the payment-gateway pools")
    assert "chg-1042" in toks and "1042" in toks and "payment-gateway" in toks and "gateway" in toks
    assert "the" not in toks and "pool" in toks


def test_light_stem():
    assert light_stem("services") == "service"
    assert light_stem("deployed") == "deploy"
    assert light_stem("queries") == "query"
    assert light_stem("process") == "process"
    assert light_stem("1042") == "1042"


def test_bm25_saturation_and_length_normalization():
    enc = BM25Encoder(k1=1.2, b=0.75, avgdl=10)
    one = enc.encode_document("timeout")
    many = enc.encode_document("timeout " * 20)
    w_one = dict(zip(one.indices, one.values, strict=True))[term_id("timeout")]
    w_many = dict(zip(many.indices, many.values, strict=True))[term_id("timeout")]
    assert w_many < (enc.k1 + 1)  # term frequency saturates
    assert w_many > w_one
    short = enc.encode_document("timeout error")
    long = enc.encode_document("timeout " + "filler " * 40)
    assert dict(zip(short.indices, short.values, strict=True))[term_id("timeout")] > \
        dict(zip(long.indices, long.values, strict=True))[term_id("timeout")]


def test_query_vector_weights_are_one():
    q = BM25Encoder().encode_query("payment payment gateway")
    assert len(q.indices) == 2 and q.values == [1.0, 1.0]


def test_rrf_rewards_agreement():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b", source="bm25"), _hit("d", source="bm25"), _hit("a", source="bm25")]
    fused = weighted_rrf([dense, sparse], k=60)
    assert [h.chunk.chunk_id for h in fused][:2] == ["b", "a"]  # in both lists -> on top
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def test_rrf_weights():
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("b", source="bm25"), _hit("a", source="bm25")]
    fused = weighted_rrf([dense, sparse], weights=[1.0, 3.0], k=60)
    assert fused[0].chunk.chunk_id == "b"


def test_mmr_prefers_diverse_results():
    hits = [_hit("a", 1.0, [1, 0]), _hit("a2", 0.99, [0.999, 0.01]), _hit("b", 0.9, [0, 1])]
    picked = [h.chunk.chunk_id for h in mmr_select(hits, k=2, lam=0.5)]
    assert picked == ["a", "b"]
    picked_relevance_only = [h.chunk.chunk_id for h in mmr_select(hits, k=2, lam=1.0)]
    assert picked_relevance_only == ["a", "a2"]


def test_hybrid_search_end_to_end(ingestion, retriever):
    ingestion.ingest_file("pool.md", b"# Pool\n\nCHG-1042 reduced the PayFlex connection pool of payment-gateway to 50.", corpus="t")
    ingestion.ingest_file("bonus.md", b"# Bonus\n\nThe bonus-engine credits free spins to players every weekend.", corpus="t")
    ingestion.ingest_file("kyc.md", b"# KYC\n\nkyc-service verifies identity documents uploaded by new players.", corpus="t")
    res = retriever.search("connection pool change CHG-1042", corpus="t")
    assert res.hits[0].chunk.doc_id == "t:pool.md"
    assert "rrf" in res.hits[0].scores
    only_bm25 = retriever.search("CHG-1042", corpus="t", config=retriever.config.with_overrides(mode="sparse"))
    assert only_bm25.hits[0].chunk.doc_id == "t:pool.md"


def test_date_filter(ingestion, retriever):
    ingestion.ingest_file("march.md", b"---\ndate: 2026-03-12\n---\n# Deposit delays\n\nDeposits were delayed.", corpus="t")
    ingestion.ingest_file("april.md", b"---\ndate: 2026-04-02\n---\n# Deposit delays again\n\nDeposits were delayed.", corpus="t")
    res = retriever.search("deposit delays", corpus="t", date_from="2026-03-01", date_to="2026-03-31")
    assert {h.chunk.doc_id for h in res.hits} == {"t:march.md"}
