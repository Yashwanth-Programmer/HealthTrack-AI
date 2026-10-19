"""Data preprocessing: raw, messy appointment rows -> cleaned PatientRecord objects.

Handles: header aliases, whitespace/casing in names, many date formats, phone
normalisation, condition aliases ("BP" -> hypertension), response-status
normalisation, duplicate rows, and rejection (with a reason) of unusable rows.
Nothing here calls the LLM, so it is fast and fully unit-testable.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from .models import PatientRecord, PreprocessResult, RejectedRecord
from .rag import KnowledgeBase

logger = logging.getLogger("healthtrack.preprocessing")

HEADER_ALIASES = {
    "patient_id": ("id", "patientid", "pid", "mrn", "patient_no"),
    "name": ("patient_name", "full_name", "patient", "fullname"),
    "age": ("patient_age",),
    "phone": ("mobile", "phone_number", "contact", "contact_number", "mobile_number"),
    "condition": ("diagnosis", "problem", "reason", "reason_for_visit", "ailment"),
    "appointment_date": ("appointment", "date", "appt_date", "visit_date", "appointment_dt"),
    "doctor": ("physician", "doctor_name", "dr", "consultant", "provider"),
    "notes": ("remarks", "instructions", "comments", "note"),
    "response_status": ("response", "status", "patient_response", "reply"),
    "attempts": ("follow_up_attempts", "previous_attempts", "attempt_count", "contact_attempts"),
}
_HEADER_LOOKUP = {alias: canon for canon, aliases in HEADER_ALIASES.items() for alias in aliases}

# Day-first first: Indian clinics write 03/04/2026 as 3 April.
DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d/%m/%y", "%d-%m-%y",
)
HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "mx"}
RESPONDED = {"responded", "replied", "reply", "yes", "y", "confirmed", "answered", "reached"}
NO_RESPONSE = {
    "no response", "no_response", "noresponse", "none", "no", "n", "not responded",
    "unreachable", "no reply", "missed", "not reachable", "did not respond",
}
MAX_NOTES_CHARS = 200
ESCALATION_ATTEMPTS = 2


class RecordError(ValueError):
    """Raised when a row cannot be turned into a usable record."""


# ---------------------------------------------------------------- field cleaners
def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]+", " ", text)).strip()


def clean_name(value: Any) -> str:
    text = re.sub(r"[^\w\s.'\-]|[\d_]", "", clean_text(value))
    text = re.sub(r"\s+", " ", text).strip(" .-'")
    # Fix ALL-CAPS / all-lowercase tokens but leave mixed-case ones ("D'Souza", "McDonald") alone.
    return " ".join(t.title() if t.isupper() or t.islower() else t for t in text.split())


def first_name(full_name: str) -> str:
    parts = full_name.split()
    while len(parts) > 1 and parts[0].lower().strip(".") in HONORIFICS:
        parts.pop(0)
    return parts[0] if parts else full_name


def clean_patient_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "", clean_text(value)).upper()


def parse_date(value: Any) -> date | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.split("T")[0] if re.match(r"^\d{4}-\d{2}-\d{2}T", text) else text
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def clean_phone(value: Any) -> str | None:
    digits = re.sub(r"\D", "", clean_text(value))
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    if 8 <= len(digits) <= 15:
        return f"+{digits}"
    return None


def mask_phone(phone: str | None) -> str:
    return "" if not phone else f"{phone[:3]}{'*' * max(len(phone) - 5, 0)}{phone[-2:]}"


def parse_age(value: Any) -> int | None:
    match = re.search(r"\d{1,3}", clean_text(value))
    if not match:
        return None
    age = int(match.group())
    return age if 0 <= age <= 120 else None


def parse_attempts(value: Any) -> int:
    match = re.search(r"\d+", clean_text(value))
    return min(int(match.group()), 20) if match else 0


def normalize_response_status(value: Any) -> str:
    text = clean_text(value).lower().replace("-", " ")
    if text in RESPONDED:
        return "responded"
    if text in NO_RESPONSE or text.replace(" ", "_") in NO_RESPONSE:
        return "no_response"
    return "pending"


def clean_doctor(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "your doctor"
    text = re.sub(r"[^\w\s.'\-]", "", text)
    match = re.match(r"(?i)^dr(?:\.\s*|\s+)(.+)$", text)
    body = match.group(1) if match else text
    if re.fullmatch(r"(?i)dr\.?", body):
        return "your doctor"
    if body.isupper() or body.islower():
        body = body.title()
    return f"Dr. {body}"


def clean_notes(value: Any) -> str:
    """Notes are untrusted text that reach the prompt: strip control chars, cap length."""
    return clean_text(value)[:MAX_NOTES_CHARS]


# ------------------------------------------------------------------ row handling
def normalise_keys(raw: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw.items():
        slug = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        canon = _HEADER_LOOKUP.get(slug, slug)
        text = clean_text(value)
        if canon not in row or not row[canon]:
            row[canon] = text
    return row


def _build_record(row: dict[str, str], number: int, kb: KnowledgeBase, today: date) -> PatientRecord:
    name = clean_name(row.get("name"))
    if not name:
        raise RecordError("missing patient name")

    raw_date = row.get("appointment_date", "")
    appt = parse_date(raw_date)
    if appt is None:
        reason = f"unreadable appointment date '{raw_date}'" if raw_date else "missing appointment date"
        raise RecordError(reason)
    if appt > today:
        raise RecordError("appointment date is in the future")

    raw_condition = row.get("condition", "")
    key = kb.resolve(raw_condition)
    if key != "general":
        label = kb.label(key)
    else:  # unknown condition: keep the clinic's own wording so the message stays accurate
        label = clean_name(raw_condition) or kb.label(key)

    status = normalize_response_status(row.get("response_status"))
    attempts = parse_attempts(row.get("attempts"))
    if status == "no_response":
        follow_up_type = "escalation" if attempts >= ESCALATION_ATTEMPTS else "reminder"
    else:
        follow_up_type = "standard"

    # A follow-up date in the past is useless, so never schedule before today.
    follow_up = max(appt + timedelta(days=kb.follow_up_days(key)), today)

    return PatientRecord(
        patient_id=clean_patient_id(row.get("patient_id")) or f"ROW{number:03d}",
        name=name,
        first_name=first_name(name),
        age=parse_age(row.get("age")),
        phone=clean_phone(row.get("phone")),
        condition_key=key,
        condition_label=label,
        appointment_date=appt,
        follow_up_date=follow_up,
        doctor=clean_doctor(row.get("doctor")),
        notes=clean_notes(row.get("notes")),
        response_status=status,
        attempts=attempts,
        follow_up_type=follow_up_type,
    )


def preprocess_records(
    rows: list[dict[str, Any]], *, kb: KnowledgeBase, today: date | None = None
) -> PreprocessResult:
    """Clean raw rows. Bad rows are rejected with a reason instead of crashing the run."""
    today = today or date.today()
    records: list[PatientRecord] = []
    rejected: list[RejectedRecord] = []
    seen: set[tuple[str, date]] = set()
    duplicates = 0

    for number, raw in enumerate(rows, start=1):
        row = normalise_keys(raw if isinstance(raw, dict) else {})
        try:
            record = _build_record(row, number, kb, today)
        except RecordError as exc:
            pid = clean_patient_id(row.get("patient_id")) or None
            rejected.append(RejectedRecord(row=number, patient_id=pid, reason=str(exc)))
            logger.warning("record rejected | row=%s | patient=%s | reason=%s", number, pid, exc)
            continue
        dedupe_key = (record.patient_id, record.appointment_date)
        if dedupe_key in seen:
            duplicates += 1
            rejected.append(RejectedRecord(row=number, patient_id=record.patient_id, reason="duplicate of an earlier row"))
            logger.warning("duplicate dropped | row=%s | patient=%s", number, record.patient_id)
            continue
        seen.add(dedupe_key)
        records.append(record)

    return PreprocessResult(records=records, rejected=rejected, duplicates_removed=duplicates)


def parse_csv(text: str) -> list[dict[str, str]]:
    """Parse CSV text into row dicts. Raises ValueError for unusable input."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")
    rows = []
    for raw in reader:
        row = {k: v for k, v in raw.items() if k is not None}
        if any(isinstance(v, str) and v.strip() for v in row.values()):
            rows.append(row)
    return rows
