"""Pydantic models shared by preprocessing, generation and the API."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

FollowUpType = Literal["standard", "reminder", "escalation"]
ResponseStatus = Literal["responded", "no_response", "pending"]


class PatientRecord(BaseModel):
    """A cleaned, validated appointment record, ready for the LLM."""

    patient_id: str
    name: str
    first_name: str
    age: int | None = None
    phone: str | None = None
    condition_key: str
    condition_label: str
    appointment_date: date
    follow_up_date: date
    doctor: str
    notes: str = ""
    response_status: ResponseStatus = "pending"
    attempts: int = 0
    follow_up_type: FollowUpType = "standard"


class RejectedRecord(BaseModel):
    row: int
    patient_id: str | None = None
    reason: str


class PreprocessResult(BaseModel):
    records: list[PatientRecord]
    rejected: list[RejectedRecord]
    duplicates_removed: int = 0


class FollowUpResult(BaseModel):
    patient_id: str
    name: str
    condition: str
    appointment_date: date
    follow_up_date: date
    follow_up_type: FollowUpType
    status: Literal["success", "fallback"]
    source: Literal["llm", "template"]
    message: str
    tips: list[str]
    error: str | None = None


class Stats(BaseModel):
    model: str
    total_input: int
    processed: int
    rejected: int
    duplicates_removed: int
    ai_generated: int
    template_generated: int
    fallback: int
    llm_retries: int = 0
    fallback_reasons: dict[str, int] = Field(default_factory=dict)
    preprocess_ms: float
    generate_ms: float
    elapsed_ms: float


class GenerateRequest(BaseModel):
    records: list[dict[str, Any]] | None = None
    csv_text: str | None = None
    model: str | None = None
    concurrency: int | None = Field(default=None, ge=1, le=50)
    batch_size: int | None = Field(default=None, ge=1, le=25)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> GenerateRequest:
        if (self.records is None) == (self.csv_text is None):
            raise ValueError("Provide exactly one of 'records' or 'csv_text'.")
        return self


class GenerateResponse(BaseModel):
    results: list[FollowUpResult]
    rejected: list[RejectedRecord]
    stats: Stats
