"""The brief's 'Could': 100 records inside 5 seconds. Latency is simulated at 1s per LLM call."""
import asyncio
import time

from conftest import TODAY, FakeLLM

from healthtrack.generator import FollowUpGenerator
from healthtrack.pipeline import run_pipeline
from healthtrack.samples import generate_sample_rows


def test_100_records_under_5_seconds_with_concurrent_batches(kb):
    llm = FakeLLM(latency=1.0)  # pessimistic: every request takes a full second
    gen = FollowUpGenerator(kb, llm)
    rows = generate_sample_rows(100, today=TODAY)
    start = time.perf_counter()
    out = asyncio.run(run_pipeline(rows, generator=gen, model="mistral-small-latest", concurrency=10, batch_size=10, today=TODAY))
    elapsed = time.perf_counter() - start
    assert out.stats.processed == 100 and out.stats.ai_generated == 100
    assert llm.calls == 10
    assert elapsed < 5, f"took {elapsed:.2f}s"


def test_sequential_calls_would_miss_the_budget(kb):
    """Documents why batching + concurrency matters: concurrency=1 serialises the calls."""
    gen = FollowUpGenerator(kb, FakeLLM(latency=0.3))
    rows = generate_sample_rows(100, today=TODAY)
    start = time.perf_counter()
    asyncio.run(run_pipeline(rows, generator=gen, model="mistral-small-latest", concurrency=1, batch_size=10, today=TODAY))
    assert time.perf_counter() - start >= 3.0


def test_offline_pipeline_is_fast(kb):
    gen = FollowUpGenerator(kb, None)
    rows = generate_sample_rows(500, today=TODAY)
    out = asyncio.run(run_pipeline(rows, generator=gen, model="offline-template", concurrency=10, batch_size=10, today=TODAY))
    assert out.stats.processed == 500 and out.stats.elapsed_ms < 2000
