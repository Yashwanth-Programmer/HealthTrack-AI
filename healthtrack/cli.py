"""Command-line runner: CSV in, one follow-up message per patient out.

    python -m healthtrack.cli data/sample_patients_100.csv
    python -m healthtrack.cli data/sample_patients_100.csv --model ministral-3b-latest --out results.csv
    python -m healthtrack.cli data/sample_patients_100.csv --model offline-template   # no API key needed
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from .config import ALL_MODELS, OFFLINE_MODEL, load_settings
from .generator import FollowUpGenerator
from .llm import LLMClient, MistralClient
from .logging_config import setup_logging
from .models import GenerateResponse
from .pipeline import run_pipeline
from .preprocessing import parse_csv
from .rag import load_knowledge_base

RESULT_COLUMNS = ["patient_id", "name", "condition", "appointment_date", "follow_up_date",
                  "follow_up_type", "source", "status", "message", "tips"]


def write_results_csv(response: GenerateResponse, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        for r in response.results:
            row = [getattr(r, c) for c in RESULT_COLUMNS]
            row[-1] = " | ".join(r.tips)
            writer.writerow(row)


async def _run(args: argparse.Namespace, llm: LLMClient | None) -> GenerateResponse:
    settings = load_settings()
    owned: MistralClient | None = None
    model = args.model or settings.effective_default_model
    if llm is None and model != OFFLINE_MODEL:
        if not settings.llm_configured:
            raise SystemExit("MISTRAL_API_KEY is not set. Add it to .env or use --model offline-template.")
        owned = llm = MistralClient(settings.api_key or "", base_url=settings.base_url,
                                    timeout=settings.request_timeout, max_retries=settings.max_retries)
        await owned.warm_up()
    try:
        generator = FollowUpGenerator(load_knowledge_base(str(settings.kb_path)), llm)
        rows = parse_csv(Path(args.csv).read_text(encoding="utf-8"))
        return await run_pipeline(rows, generator=generator, model=model,
                                  concurrency=args.concurrency or settings.concurrency,
                                  batch_size=args.batch_size or settings.batch_size)
    finally:
        if owned:
            await owned.aclose()


def main(argv: list[str] | None = None, *, llm: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="path to the patient appointments CSV")
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", type=Path, help="also write results to this CSV file")
    parser.add_argument("--quiet", action="store_true", help="print only the summary")
    args = parser.parse_args(argv)

    setup_logging(load_settings().log_dir, console=False)
    response = asyncio.run(_run(args, llm))

    if not args.quiet:
        for r in response.results:
            tag = "AI" if r.source == "llm" else ("FALLBACK" if r.status == "fallback" else "TEMPLATE")
            print(f"[{r.patient_id}] {r.name} | {r.condition} | {r.follow_up_type} | {tag}\n    {r.message}\n")
    s = response.stats
    for rej in response.rejected:
        print(f"REJECTED row {rej.row} ({rej.patient_id or '?'}): {rej.reason}", file=sys.stderr)
    print(f"{s.processed}/{s.total_input} messages | AI {s.ai_generated} | template {s.template_generated} | "
          f"fallback {s.fallback} | rejected {s.rejected} | {s.elapsed_ms / 1000:.2f}s | model {s.model}")
    if args.out:
        write_results_csv(response, args.out)
        print(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
