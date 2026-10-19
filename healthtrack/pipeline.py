"""End-to-end pipeline: raw rows -> preprocessing -> RAG + LLM generation -> response."""
from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import date
from typing import Any

from .generator import FollowUpGenerator
from .models import GenerateResponse, Stats
from .preprocessing import preprocess_records

logger = logging.getLogger("healthtrack.pipeline")


async def run_pipeline(
    rows: list[dict[str, Any]],
    *,
    generator: FollowUpGenerator,
    model: str,
    concurrency: int,
    batch_size: int,
    today: date | None = None,
) -> GenerateResponse:
    retries_before = getattr(generator.llm, "retry_count", 0)
    t0 = time.perf_counter()
    prep = preprocess_records(rows, kb=generator.kb, today=today)
    t1 = time.perf_counter()
    results = await generator.generate(prep.records, model=model, concurrency=concurrency, batch_size=batch_size)
    t2 = time.perf_counter()

    stats = Stats(
        model=model,
        total_input=len(rows),
        processed=len(results),
        rejected=len(prep.rejected),
        duplicates_removed=prep.duplicates_removed,
        ai_generated=sum(r.source == "llm" for r in results),
        template_generated=sum(r.source == "template" and r.status == "success" for r in results),
        fallback=sum(r.status == "fallback" for r in results),
        llm_retries=getattr(generator.llm, "retry_count", 0) - retries_before,
        fallback_reasons=dict(Counter((r.error or 'unknown')[:160] for r in results if r.status == 'fallback')),
        preprocess_ms=round((t1 - t0) * 1000, 1),
        generate_ms=round((t2 - t1) * 1000, 1),
        elapsed_ms=round((t2 - t0) * 1000, 1),
    )
    logger.info(
        "run complete | model=%s | input=%s | processed=%s | rejected=%s | ai=%s | fallback=%s | retries=%s | elapsed_ms=%s",
        model, stats.total_input, stats.processed, stats.rejected, stats.ai_generated, stats.fallback, stats.llm_retries, stats.elapsed_ms,
    )
    return GenerateResponse(results=results, rejected=prep.rejected, stats=stats)
