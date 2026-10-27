"""Index a corpus into Qdrant (local embedded mode or Qdrant Cloud, depending on .env).

Examples:
  python -m scripts.index_corpus --corpus incidents                      # Acme Play files
  python -m scripts.index_corpus --corpus incidents --strategy fixed     # same files, other chunker
  python -m scripts.index_corpus --corpus hotpotqa                       # ~66k Wikipedia paragraphs
  python -m scripts.index_corpus --corpus incidents --reset              # drop collections first
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.config import DATA_DIR, get_settings
from app.index.bm25 import tokenize
from app.index.embeddings import CachedPassageEmbedder
from app.ingestion.metadata import extract_entities
from app.ingestion.pipeline import make_doc_id
from app.models import Document
from app.observability import setup_logging
from app.services import Services

INCIDENTS_DIR = DATA_DIR / "corpus" / "incidents"
HOTPOT_DOCS = DATA_DIR / "processed" / "hotpotqa_docs.jsonl"
EMBED_CACHE = DATA_DIR / "cache" / "hotpotqa_embeddings.npz"  # default model (bge-small)
HOTPOT_QUESTIONS = DATA_DIR.parent / "eval" / "datasets" / "hotpotqa_questions.jsonl"


def embed_cache_path(model: str) -> Path:
    """One cache file per embedding model - vectors of different models must never be mixed."""
    if model == "BAAI/bge-small-en-v1.5":
        return EMBED_CACHE
    return DATA_DIR / "cache" / f"hotpotqa_embeddings_{model.replace('/', '_')}.npz"


def embedding_dim(model: str) -> int:
    from fastembed import TextEmbedding

    return next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model)


def index_incidents(svc: Services, force: bool = False) -> None:
    files = sorted(p for p in INCIDENTS_DIR.rglob("*") if p.is_file())
    if not files:
        raise SystemExit("No corpus files. Run: python -m scripts.generate_incident_corpus")
    counts: dict[str, int] = {}
    lengths = []
    for path in files:
        if force:  # re-ingest from scratch (e.g. after adding a metadata field); only this corpus is touched
            svc.ingestion.delete(make_doc_id("incidents", path.name))
        res = svc.ingestion.ingest_file(path.name, path.read_bytes(), corpus="incidents", protected=True)
        counts[res.status] = counts.get(res.status, 0) + 1
        extra = f" duplicate_of={res.duplicate_of}" if res.duplicate_of else ""
        extra += f" near_duplicates={res.near_duplicates}" if res.near_duplicates else ""
        print(f"  {res.status:9s} {res.chunks:3d} chunks  {path.relative_to(INCIDENTS_DIR)}{extra}")
        if res.status in ("added", "updated"):
            lengths += [len(tokenize(c.context_text())) for c in svc.store.get_doc_chunks(res.doc_id)]
    print(f"incidents: {counts}; avg chunk length {sum(lengths) / max(len(lengths), 1):.1f} tokens")


def index_hotpotqa(svc: Services, limit: int | None, embed_only: bool = False, subset: int | None = None) -> None:
    if not HOTPOT_DOCS.exists():
        raise SystemExit("Missing HotpotQA docs. Run: python -m scripts.prepare_hotpotqa")
    docs = []
    keep: set[str] | None = None
    if subset:  # gold paragraphs of the evaluation questions + `subset` random other paragraphs (seed 42)
        import random

        gold = {g for line in HOTPOT_QUESTIONS.open() for g in json.loads(line)["gold_doc_ids"]}
        all_ids = [json.loads(line)["doc_id"] for line in HOTPOT_DOCS.open()]
        others = [d for d in all_ids if d not in gold]
        keep = gold | set(random.Random(42).sample(others, subset))
        print(f"subset: {len(gold)} gold + {subset} other paragraphs")
    with HOTPOT_DOCS.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            if keep is not None and d["doc_id"] not in keep:
                continue
            docs.append(Document(
                doc_id=d["doc_id"], title=d["title"], text=d["text"], corpus="hotpotqa", source="hotpot_qa/distractor/validation",
                format="jsonl", metadata={"doc_type": "encyclopedia", "entities": extract_entities(d["text"])},
            ))
    started = time.time()
    cache = CachedPassageEmbedder(svc.embedder, str(embed_cache_path(svc.settings.embedding_model)))
    print(f"embedding cache: {len(cache)} vectors already computed")
    svc.__dict__["embedder"] = cache  # the ingestion service below picks up the cached embedder
    svc.__dict__.pop("ingestion", None)
    try:
        if embed_only:
            chunks = [c for d in docs for c in svc.ingestion.make_chunks(d, version=1)]
            for i, start in enumerate(range(0, len(chunks), 2048), start=1):
                cache.embed_passages([c.context_text() for c in chunks[start:start + 2048]], batch_size=32)
                print(f"  embedded {min(start + 2048, len(chunks))}/{len(chunks)} chunks "
                      f"({time.time() - started:.0f}s)", flush=True)
                if i % 4 == 0:
                    cache.save()  # checkpoint: an interrupted run resumes from here
            print(f"avg chunk length {sum(len(tokenize(c.context_text())) for c in chunks) / len(chunks):.1f} tokens")
            return
        stats = svc.ingestion.ingest_bulk(docs, batch_size=512, protected=True, detect_near_duplicates=True)
        print(f"hotpotqa: {stats} in {time.time() - started:.0f}s")
    finally:
        cache.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["incidents", "hotpotqa"], required=True)
    parser.add_argument("--strategy", choices=["structure", "fixed"], default=None)
    parser.add_argument("--reset", action="store_true", help="delete this strategy's collections first (ALL corpora)")
    parser.add_argument("--force", action="store_true", help="incidents: re-ingest every file even if unchanged")
    parser.add_argument("--limit", type=int, default=None, help="index only the first N HotpotQA docs")
    parser.add_argument("--embed-only", action="store_true", help="HotpotQA: only compute and cache embeddings")
    parser.add_argument("--subset", type=int, default=None, help="HotpotQA: eval gold paragraphs + N random others")
    parser.add_argument("--embedding-model", default=None, help="e.g. BAAI/bge-base-en-v1.5 (for model comparison)")
    parser.add_argument("--prefix", default=None, help="collection prefix, to keep comparison indexes separate")
    args = parser.parse_args()

    setup_logging("INFO")
    logging.getLogger("app.ingestion.pipeline").setLevel(logging.INFO)
    settings = get_settings()
    if args.strategy:
        settings = settings.model_copy(update={"chunk_strategy": args.strategy})
    if args.embedding_model:
        settings = settings.model_copy(update={"embedding_model": args.embedding_model,
                                               "embedding_dim": embedding_dim(args.embedding_model)})
    if args.prefix:
        settings = settings.model_copy(update={"collection_prefix": args.prefix})
    svc = Services(settings)
    print(f"Qdrant: {'cloud ' + settings.qdrant_url if settings.qdrant_url else 'local ' + settings.qdrant_path}; "
          f"collection {svc.store.chunks_collection}")
    if args.reset:
        svc.store.reset()
    if args.corpus == "incidents":
        index_incidents(svc, args.force)
    else:
        index_hotpotqa(svc, args.limit, args.embed_only, args.subset)
    print(f"total chunks in collection: {svc.store.count_chunks()}")


if __name__ == "__main__":
    Path(DATA_DIR).mkdir(exist_ok=True)
    main()
