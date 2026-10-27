"""Concurrent load test of the search API (no LLM calls, so it costs no LLM quota).

Sends the HotpotQA evaluation questions to GET /api/search against the full 66k-document index at
increasing concurrency and reports latency percentiles, throughput and errors.

Usage:
  .venv/bin/python -m uvicorn app.api.main:app --port 8001      # in another terminal
  .venv/bin/python -m scripts.load_test --url http://localhost:8001 --levels 1,4,8,16 --requests 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "datasets" / "hotpotqa_questions.jsonl"


async def run_level(client: httpx.AsyncClient, url: str, questions: list[str], concurrency: int, n: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one(q: str) -> None:
        nonlocal errors
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.get(f"{url}/api/search", params={"q": q, "corpus": "hotpotqa", "k": 8})
                if r.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - t) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(one(questions[i % len(questions)]) for i in range(n)))
    wall = time.perf_counter() - started
    lat = sorted(latencies)
    return {
        "concurrency": concurrency, "requests": n, "errors": errors,
        "p50_ms": round(statistics.median(lat)), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))]),
        "max_ms": round(lat[-1]), "throughput_rps": round(n / wall, 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--levels", default="1,4,8,16")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "load_test.json"))
    args = parser.parse_args()

    questions = [json.loads(line)["question"] for line in QUESTIONS.open()]
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        health = (await client.get(f"{args.url}/api/health")).json()
        await client.get(f"{args.url}/api/search", params={"q": "warm up", "corpus": "hotpotqa"})  # load models
        for level in (int(x) for x in args.levels.split(",")):
            res = await run_level(client, args.url, questions, level, args.requests)
            results.append(res)
            print(res, flush=True)
    Path(args.out).write_text(json.dumps({"health": health, "results": results}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
