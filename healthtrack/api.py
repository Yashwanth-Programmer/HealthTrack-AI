"""FastAPI app: JSON API + the single-page UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .config import ALL_MODELS, BASE_DIR, OFFLINE_MODEL, Settings, load_settings
from .generator import FollowUpGenerator
from .llm import LLMClient, MistralClient
from .logging_config import setup_logging
from .middleware import install_middleware
from .models import GenerateRequest, GenerateResponse
from .pipeline import run_pipeline
from .preprocessing import parse_csv
from .rag import KnowledgeBase, load_knowledge_base
from .samples import generate_sample_rows, rows_to_csv

logger = logging.getLogger("healthtrack.api")
INDEX_HTML = BASE_DIR / "static" / "index.html"


def create_app(
    *, llm: LLMClient | None = None, kb: KnowledgeBase | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_dir)
    knowledge_base = kb or load_knowledge_base(str(settings.kb_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = llm
        owned: MistralClient | None = None
        if client is None and settings.llm_configured:
            owned = MistralClient(
                settings.api_key or "", base_url=settings.base_url,
                timeout=settings.request_timeout, max_retries=settings.max_retries,
            )
            ok = await owned.warm_up()
            logger.info("Mistral client ready | warm_up_ok=%s", ok)
            client = owned
        elif client is None:
            logger.warning("MISTRAL_API_KEY not set: only the offline-template model is available.")
        app.state.generator = FollowUpGenerator(knowledge_base, client, log_message_text=settings.log_message_text)
        yield
        if owned:
            await owned.aclose()

    app = FastAPI(title="HealthTrack Follow-Up", version=__version__, lifespan=lifespan)
    install_middleware(app, rate_limit_per_minute=settings.rate_limit_per_minute)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "-")
        logger.exception("unhandled error | request_id=%s", rid)
        return JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @app.get("/healthz", tags=["ops"])
    async def liveness() -> dict:
        """Liveness probe: the process is up and serving."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readiness(request: Request) -> dict:
        """Readiness probe: knowledge base loaded and generator initialised."""
        generator: FollowUpGenerator = request.app.state.generator
        return {
            "status": "ready", "version": __version__,
            "llm_configured": generator.llm is not None,
            "knowledge_base_conditions": len(generator.kb.conditions),
        }

    @app.get("/api/config")
    async def config(request: Request) -> dict:
        configured = request.app.state.generator.llm is not None
        return {
            "llm_configured": configured,
            "models": list(ALL_MODELS) if configured else [OFFLINE_MODEL],
            "default_model": settings.effective_default_model if configured else OFFLINE_MODEL,
            "concurrency": settings.concurrency,
            "batch_size": settings.batch_size,
            "max_records": settings.max_records,
        }

    @app.get("/api/sample")
    async def sample(n: int = Query(100, ge=1, le=settings.max_records)) -> dict:
        return {"records": generate_sample_rows(n)}

    @app.get("/api/sample.csv")
    async def sample_csv(n: int = Query(100, ge=1, le=settings.max_records)) -> Response:
        return Response(
            rows_to_csv(generate_sample_rows(n)), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="patients_{n}.csv"'},
        )

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        generator: FollowUpGenerator = request.app.state.generator
        model = body.model or (settings.effective_default_model if generator.llm else OFFLINE_MODEL)
        if model not in ALL_MODELS:
            raise HTTPException(422, f"Unknown model '{model}'.")
        if model != OFFLINE_MODEL and generator.llm is None:
            raise HTTPException(400, "No MISTRAL_API_KEY configured on the server; choose 'offline-template'.")

        if body.csv_text is not None:
            if len(body.csv_text.encode("utf-8")) > settings.max_csv_bytes:
                raise HTTPException(413, "CSV is too large.")
            try:
                rows = parse_csv(body.csv_text)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        else:
            rows = body.records or []

        if not rows:
            raise HTTPException(422, "No records to process.")
        if len(rows) > settings.max_records:
            raise HTTPException(413, f"Too many records ({len(rows)}); the limit is {settings.max_records}.")

        return await run_pipeline(
            rows, generator=generator, model=model,
            concurrency=body.concurrency or settings.concurrency,
            batch_size=body.batch_size or settings.batch_size,
        )

    return app
