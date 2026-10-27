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
EMBED_CACHE = DATA_DIR / "cache" / "hotpotqa_embeddings.npz"


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


def index_hotpotqa(svc: Services, limit: int | None, embed_only: bool = False) -> None:
    if not HOTPOT_DOCS.exists():
        raise SystemExit("Missing HotpotQA docs. Run: python -m scripts.prepare_hotpotqa")
    docs = []
    with HOTPOT_DOCS.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            docs.append(Document(
                doc_id=d["doc_id"], title=d["title"], text=d["text"], corpus="hotpotqa", source="hotpot_qa/distractor/validation",
                format="jsonl", metadata={"doc_type": "encyclopedia", "entities": extract_entities(d["text"])},
            ))
    started = time.time()
    cache = CachedPassageEmbedder(svc.embedder, str(EMBED_CACHE))
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
    args = parser.parse_args()

    setup_logging("INFO")
    logging.getLogger("app.ingestion.pipeline").setLevel(logging.INFO)
    settings = get_settings()
    if args.strategy:
        settings = settings.model_copy(update={"chunk_strategy": args.strategy})
    svc = Services(settings)
    print(f"Qdrant: {'cloud ' + settings.qdrant_url if settings.qdrant_url else 'local ' + settings.qdrant_path}; "
          f"collection {svc.store.chunks_collection}")
    if args.reset:
        svc.store.reset()
    if args.corpus == "incidents":
        index_incidents(svc, args.force)
    else:
        index_hotpotqa(svc, args.limit, args.embed_only)
    print(f"total chunks in collection: {svc.store.count_chunks()}")


if __name__ == "__main__":
    Path(DATA_DIR).mkdir(exist_ok=True)
    main()
