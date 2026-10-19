"""Cross-cutting HTTP concerns: request ids, access log, security headers, rate limiting."""
from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_config import request_id_var

access_log = logging.getLogger("healthtrack.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
RATE_LIMITED_PATH = "/api/generate"
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")

CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


class SlidingWindowLimiter:
    """In-memory per-client limiter. Fine for one process; use Redis/gateway when scaling out."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> float:
        """Return 0.0 if the request is allowed, else the seconds until it would be."""
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window - (now - hits[0])
        hits.append(now)
        if len(self._hits) > 10_000:  # bound memory under address churn
            self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
        return 0.0


def install_middleware(app: FastAPI, *, rate_limit_per_minute: int) -> SlidingWindowLimiter | None:
    limiter = SlidingWindowLimiter(rate_limit_per_minute) if rate_limit_per_minute > 0 else None

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        try:
            wait = 0.0
            if limiter and request.method == "POST" and request.url.path == RATE_LIMITED_PATH:
                wait = limiter.check(request.client.host if request.client else "unknown")
            if wait > 0:
                response = JSONResponse(
                    {"detail": "Rate limit exceeded. Please retry shortly."}, status_code=429,
                    headers={"Retry-After": str(max(1, math.ceil(wait)))},
                )
            else:
                response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            if request.url.path.startswith("/api/"):
                response.headers["Cache-Control"] = "no-store"
            if not request.url.path.startswith(DOC_PATHS):
                response.headers["Content-Security-Policy"] = CSP
            return response
        finally:
            access_log.info("%s %s -> %s in %.0fms", request.method, request.url.path, status,
                            (time.perf_counter() - started) * 1000)
            request_id_var.reset(token)

    return limiter
