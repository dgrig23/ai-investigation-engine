"""FastAPI application: REST API + streaming endpoint + the web UI.

Endpoints (all JSON unless noted):
  GET    /api/health                 liveness + dependency status
  GET    /api/stats                  document/chunk counts per corpus
  GET    /api/metrics                latency percentiles, errors, LLM usage
  GET    /api/documents              list documents (paginated)
  GET    /api/documents/{doc_id}     one document with its chunks
  POST   /api/documents              upload one or more files (multipart)
  PUT    /api/documents/{doc_id}     replace a document's content (new version)
  DELETE /api/documents/{doc_id}     delete a document and its chunks
  GET    /api/search                 raw hybrid search with per-stage scores (debugging)
  POST   /api/ask                    investigate a question, return the full result
  POST   /api/ask/stream             same, as Server-Sent Events (live agent trace)
  GET    /                           web UI
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.ingestion.loaders import SUPPORTED_EXTENSIONS, DocumentParseError, UnsupportedFormatError
from app.observability import metrics, setup_logging
from app.services import Services, get_services

setup_logging()
log = logging.getLogger("api")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PROTECTED_CORPORA = {"incidents", "hotpotqa"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models in a background thread at startup so the first request is not slow."""
    def _load():
        try:
            svc = get_services()
            _ = svc.store, svc.embedder, svc.reranker
            svc.embedder.embed_query("warm up")
            log.info("models loaded")
        except Exception:
            log.exception("warm-up failed")
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="AI Investigation Engine", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------ middleware / helpers


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
        response = JSONResponse({"detail": "Internal server error", "request_id": request_id}, status_code=500)
    latency = (time.perf_counter() - started) * 1000
    route = request.scope.get("route")
    metrics.observe(getattr(route, "path", request.url.path), response.status_code, latency)
    response.headers["x-request-id"] = request_id
    if request.url.path.startswith("/api"):
        log.info("request", extra={"request_id": request_id, "path": request.url.path,
                                   "status": response.status_code, "latency_ms": round(latency, 1)})
    return response


class RateLimiter:
    """Sliding-window limiter per client IP: protects the free LLM quota on the public demo."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, f"Rate limit: at most {self.per_minute} questions per minute. Please wait.")
            q.append(now)


class AnswerCache:
    """Small LRU cache of finished answers (same question + corpus + mode -> instant reply)."""

    def __init__(self, size: int = 256):
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: str, value: dict) -> None:
        if value.get("status") == "llm_unavailable":
            return  # never cache degraded answers
        with self._lock:
            self._data[key] = value
            if len(self._data) > self._size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_limiter: RateLimiter | None = None
answer_cache = AnswerCache()


def services() -> Services:
    return get_services()


def limiter(svc: Services = Depends(services)) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(svc.settings.ask_rate_limit_per_minute)
    return _limiter


def require_admin_for(corpus: str, token: str | None, svc: Services) -> None:
    if corpus not in PROTECTED_CORPORA:
        return
    expected = svc.settings.admin_token.get_secret_value() if svc.settings.admin_token else None
    if not expected or token != expected:
        raise HTTPException(403, f"The '{corpus}' corpus is read-only on the public demo. Use the 'uploads' corpus.")


# ------------------------------------------------------------------ system


@app.get("/api/health")
def health(svc: Services = Depends(services)) -> dict[str, Any]:
    try:
        svc.store.client.get_collections()
        db = "ok"
    except Exception as exc:  # report, don't crash
        db = f"error: {type(exc).__name__}"
    return {"status": "ok" if db == "ok" else "degraded", "vector_db": db,
            "vector_db_mode": "local" if svc.store.is_local else "cloud",
            "llm_providers": svc.llm.describe(), "chunk_strategy": svc.settings.chunk_strategy}


@app.get("/api/stats")
def stats(svc: Services = Depends(services)) -> dict[str, Any]:
    corpora = {}
    for corpus in ("incidents", "hotpotqa", "uploads"):
        corpora[corpus] = {"documents": svc.store.count_documents(corpus), "chunks": svc.store.count_chunks(corpus)}
    return {"corpora": corpora, "supported_formats": SUPPORTED_EXTENSIONS}


@app.get("/api/metrics")
def get_metrics(svc: Services = Depends(services)) -> dict[str, Any]:
    return {**metrics.snapshot(), "llm": svc.llm.stats.as_dict()}


# ------------------------------------------------------------------ documents


@app.get("/api/documents")
def list_documents(corpus: str | None = None, limit: int = Query(50, le=200), offset: str | None = None,
                   svc: Services = Depends(services)) -> dict[str, Any]:
    docs, next_offset = svc.store.list_documents(corpus=corpus, limit=limit, offset=offset)
    for d in docs:
        d.pop("lsh", None)
    return {"documents": docs, "next_offset": next_offset}


@app.get("/api/documents/{doc_id:path}")
def get_document(doc_id: str, svc: Services = Depends(services)) -> dict[str, Any]:
    doc = svc.store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    doc.pop("lsh", None)
    chunks = svc.store.get_doc_chunks(doc_id)
    return {"document": doc, "chunks": [{"chunk_id": c.chunk_id, "position": c.position, "section": c.section,
                                         "text": c.text, "entities": c.metadata.get("entities", [])} for c in chunks]}


async def _read_upload(file: UploadFile, max_mb: int) -> bytes:
    data = await file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File '{file.filename}' is larger than {max_mb} MB")
    return data


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...), corpus: str = Form("uploads"),
                           x_admin_token: str | None = Header(None), svc: Services = Depends(services)):
    require_admin_for(corpus, x_admin_token, svc)
    results = []
    for file in files:
        data = await _read_upload(file, svc.settings.max_upload_mb)
        try:
            res = svc.ingestion.ingest_file(file.filename or "upload.txt", data, corpus=corpus)
            results.append({**res.__dict__, "filename": file.filename})
        except (UnsupportedFormatError, DocumentParseError) as exc:
            results.append({"filename": file.filename, "status": "error", "error": str(exc)})
    answer_cache.clear()
    return {"results": results}


@app.put("/api/documents/{doc_id:path}")
async def update_document(doc_id: str, file: UploadFile = File(...), x_admin_token: str | None = Header(None),
                          svc: Services = Depends(services)):
    existing = svc.store.get_document(doc_id)
    if not existing:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    require_admin_for(existing["corpus"], x_admin_token, svc)
    data = await _read_upload(file, svc.settings.max_upload_mb)
    try:
        res = svc.ingestion.ingest_file(file.filename or existing["source"], data, corpus=existing["corpus"], doc_id=doc_id)
    except (UnsupportedFormatError, DocumentParseError) as exc:
        raise HTTPException(422, str(exc)) from exc
    answer_cache.clear()
    return res.__dict__


@app.delete("/api/documents/{doc_id:path}")
def delete_document(doc_id: str, x_admin_token: str | None = Header(None), svc: Services = Depends(services)):
    existing = svc.store.get_document(doc_id)
    if not existing:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    require_admin_for(existing["corpus"], x_admin_token, svc)
    svc.ingestion.delete(doc_id)
    answer_cache.clear()
    return {"deleted": doc_id}


# ------------------------------------------------------------------ search & ask


@app.get("/api/search")
def search(q: str = Query(..., min_length=2, max_length=500), corpus: str | None = None,
           mode: str = Query("hybrid", pattern="^(dense|sparse|hybrid)$"), k: int = Query(8, ge=1, le=30),
           rerank: bool = True, mmr: bool = True,
           doc_type: str | None = None,
           date_from: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
           date_to: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
           svc: Services = Depends(services)) -> dict[str, Any]:
    cfg = svc.retrieval_config.with_overrides(mode=mode, final_k=k, use_reranker=rerank, use_mmr=mmr)
    res = svc.retriever.search(q, corpus=corpus or None,
                               doc_types=[doc_type] if doc_type else None,
                               date_from=date_from, date_to=date_to, config=cfg)
    return {"query": q, "timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
            "best_rerank_score": res.best_rerank_score,
            "hits": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title,
                      "section": h.chunk.section, "text": h.chunk.text, "doc_type": h.chunk.metadata.get("doc_type"),
                      "date": h.chunk.metadata.get("date"), "score": round(h.score, 4),
                      "scores": {k: round(v, 4) for k, v in h.scores.items()}} for h in res.hits]}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    corpus: str | None = "incidents"
    mode: str = Field("agent", pattern="^(agent|single_shot)$")
    max_steps: int | None = Field(None, ge=1, le=6)


def _agent_config(req: AskRequest, svc: Services):
    cfg = replace(svc.agent_config, mode=req.mode)
    return replace(cfg, max_steps=req.max_steps) if req.max_steps else cfg


def _cache_key(req: AskRequest) -> str:
    return json.dumps([req.question.strip().lower(), req.corpus, req.mode, req.max_steps])


@app.post("/api/ask")
def ask(req: AskRequest, request: Request, svc: Services = Depends(services), rl: RateLimiter = Depends(limiter)):
    if cached := answer_cache.get(_cache_key(req)):
        metrics.incr("answer_cache_hits")
        return {**cached, "cached": True}
    rl.check(request.client.host if request.client else "unknown")
    result = svc.investigator.run(req.question, corpus=req.corpus or None, config=_agent_config(req, svc))
    metrics.incr(f"answers_{result.get('status', 'unknown')}")
    answer_cache.put(_cache_key(req), result)
    return result


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest, request: Request, svc: Services = Depends(services), rl: RateLimiter = Depends(limiter)):
    cached = answer_cache.get(_cache_key(req))
    if not cached:
        rl.check(request.client.host if request.client else "unknown")

    def events():
        if cached:
            yield f"data: {json.dumps({'type': 'final', 'result': {**cached, 'cached': True}})}\n\n"
            return
        try:
            for event in svc.investigator.stream(req.question, corpus=req.corpus or None, config=_agent_config(req, svc)):
                if event["type"] == "final":
                    answer_cache.put(_cache_key(req), event["result"])
                    metrics.incr(f"answers_{event['result'].get('status', 'unknown')}")
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            log.exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'[:300]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ UI


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
