"""Lightweight observability: structured logs, request ids and in-process metrics.

`/api/metrics` exposes request counts, error counts, latency percentiles per route and LLM usage.
No external service is needed (fits free hosting); the JSON is easy to scrape later.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(time.time(), 3), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for key in ("request_id", "path", "status", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Metrics:
    def __init__(self, window: int = 500):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.events: dict[str, int] = defaultdict(int)

    def observe(self, route: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests[route] += 1
            if status >= 500:
                self.errors[route] += 1
            self.latencies[route].append(latency_ms)

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.events[name] += value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            routes = {}
            for route, lat in self.latencies.items():
                arr = np.array(lat) if lat else np.array([0.0])
                routes[route] = {"requests": self.requests[route], "errors": self.errors[route],
                                 "p50_ms": round(float(np.percentile(arr, 50)), 1),
                                 "p95_ms": round(float(np.percentile(arr, 95)), 1)}
            return {"uptime_s": round(time.time() - self.started), "routes": routes, "events": dict(self.events)}


metrics = Metrics()
