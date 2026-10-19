"""Async Mistral client with a hard latency budget, retries and a swappable interface."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Protocol

import httpx

logger = logging.getLogger("healthtrack.llm")
RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure to obtain a usable completion. The message never contains secrets."""


class LLMClient(Protocol):
    async def complete(self, *, model: str, messages: list[dict], max_tokens: int) -> str: ...


class MistralClient:
    """Thin wrapper over POST /v1/chat/completions.

    `timeout` is a total budget per call: attempts and back-off sleeps all share it,
    so a slow or rate-limited request can never stall the whole batch run.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 15.0,
        max_retries: int = 2,
        http: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._timeout = timeout
        self._max_retries = max_retries
        self.retry_count = 0  # rate-limit/5xx retries so far; surfaced in run stats
        self._http = http or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
        )

    async def warm_up(self) -> bool:
        """Open the TLS connection (and sanity-check the key) before the first real call."""
        try:
            resp = await self._http.get(f"{self._base_url}/models", headers=self._headers, timeout=4.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, *, model: str, messages: list[dict], max_tokens: int) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        deadline = time.monotonic() + self._timeout
        last_error = "no attempt made"

        for attempt in range(self._max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            delay = 0.3 * (2**attempt)
            try:
                resp = await self._http.post(self._url, json=payload, headers=self._headers, timeout=remaining)
            except httpx.HTTPError as exc:
                last_error = f"network error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    return self._extract(resp)
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUSES:
                    raise LLMError(last_error)
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    delay = min(float(retry_after), 1.5)
            delay += random.uniform(0, 0.15)
            if time.monotonic() + delay >= deadline:
                break
            self.retry_count += 1
            logger.info("LLM retry | attempt=%s | reason=%s | sleep=%.2fs", attempt + 1, last_error, delay)
            await asyncio.sleep(delay)

        raise LLMError(f"{last_error}; gave up within {self._timeout:.1f}s budget")

    @staticmethod
    def _extract(resp: httpx.Response) -> str:
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape from LLM") from exc
        if isinstance(content, list):  # some models return content chunks
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty completion from LLM")
        return content
