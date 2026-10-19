"""Runtime configuration, read from environment variables (a .env file is supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OFFLINE_MODEL = "offline-template"
MISTRAL_MODELS = (
    "mistral-small-latest",
    "ministral-8b-latest",
    "ministral-3b-latest",
    "open-mistral-nemo",
)
ALL_MODELS = (*MISTRAL_MODELS, OFFLINE_MODEL)


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        return default
    return max(low, min(high, value))


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    default_model: str
    concurrency: int
    batch_size: int
    request_timeout: float
    max_retries: int
    max_records: int
    max_csv_bytes: int
    rate_limit_per_minute: int
    log_message_text: bool
    log_dir: Path
    kb_path: Path

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def effective_default_model(self) -> str:
        """Model preselected in the UI: the real one if a key exists, else offline."""
        return self.default_model if self.llm_configured else OFFLINE_MODEL


def load_settings() -> Settings:
    model = os.getenv("MISTRAL_MODEL", MISTRAL_MODELS[0]).strip()
    if model not in MISTRAL_MODELS:
        model = MISTRAL_MODELS[0]
    return Settings(
        api_key=os.getenv("MISTRAL_API_KEY", "").strip() or None,
        base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/"),
        default_model=model,
        concurrency=_env_int("LLM_CONCURRENCY", 10, low=1, high=50),
        batch_size=_env_int("LLM_BATCH_SIZE", 10, low=1, high=25),
        request_timeout=_env_float("LLM_TIMEOUT_SECONDS", 15.0, low=1.0, high=60.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 2, low=0, high=5),
        max_records=_env_int("MAX_RECORDS", 500, low=1, high=5000),
        max_csv_bytes=_env_int("MAX_CSV_BYTES", 2_000_000, low=1_000, high=20_000_000),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 30, low=0, high=10_000),
        log_message_text=os.getenv("LOG_MESSAGE_TEXT", "true").strip().lower() not in {"0", "false", "no"},
        log_dir=Path(os.getenv("LOG_DIR", BASE_DIR / "logs")),
        kb_path=Path(os.getenv("KB_PATH", BASE_DIR / "data" / "knowledge_base.json")),
    )
