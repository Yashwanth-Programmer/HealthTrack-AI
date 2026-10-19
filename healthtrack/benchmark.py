"""Find the fastest reliable settings for YOUR Mistral key.

    python -m healthtrack.benchmark                      # one run with .env defaults
    python -m healthtrack.benchmark --model ministral-3b-latest --batch 10 --concurrency 10
    python -m healthtrack.benchmark --sweep              # try several combinations
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import MISTRAL_MODELS, load_settings
from .generator import FollowUpGenerator
from .llm import MistralClient
from .pipeline import run_pipeline
from .rag import load_knowledge_base
from .samples import generate_sample_rows

SWEEP_MODELS = ("ministral-3b-latest", "ministral-8b-latest", "mistral-small-latest")
SWEEP_SHAPES = ((5, 20), (10, 10), (20, 5))  # (batch_size, concurrency)


async def _main(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.llm_configured:
        print("MISTRAL_API_KEY is not set. Put it in .env first.")
        return 1
    client = MistralClient(
        settings.api_key or "", base_url=settings.base_url,
        timeout=settings.request_timeout, max_retries=settings.max_retries,
    )
    print(f"warm-up ok: {await client.warm_up()} | per-request budget: {settings.request_timeout}s\n")
    generator = FollowUpGenerator(load_knowledge_base(str(settings.kb_path)), client)
    rows = generate_sample_rows(args.records)

    if args.sweep:
        combos = [(m, b, c) for m in SWEEP_MODELS for b, c in SWEEP_SHAPES]
    else:
        combos = [(args.model or settings.default_model, args.batch or settings.batch_size,
                   args.concurrency or settings.concurrency)]

    print(f"{'model':<24}{'batch':>6}{'conc':>6}{'time(s)':>9}{'AI':>5}{'fallback':>10}  verdict")
    for model, batch, conc in combos:
        out = await run_pipeline(rows, generator=generator, model=model, concurrency=conc, batch_size=batch)
        s = out.stats
        ok = s.elapsed_ms < 5000 and s.fallback == 0
        print(f"{model:<24}{batch:>6}{conc:>6}{s.elapsed_ms / 1000:>9.2f}{s.ai_generated:>5}{s.fallback:>10}  "
              f"{'PASS' if ok else 'FAIL'}")
        for reason, n in s.fallback_reasons.items():
            print(f"    fallback x{n}: {reason}")
        await asyncio.sleep(2)  # let rate-limit windows recover between runs
    await client.aclose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MISTRAL_MODELS)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="try several models and batch shapes")
    sys.exit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
