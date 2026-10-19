"""Message generation: RAG retrieval + batched, concurrent LLM calls + safety nets.

Design for speed: patients are grouped into small batches (default 10) and the
batches are sent to the LLM concurrently, so 100 records cost ~10 parallel calls
instead of 100 sequential ones.

Design for reliability, per patient:
  * LLM failed / timed out / omitted the patient  -> deterministic template (status="fallback")
  * LLM message lacks the retrieved health tip    -> tip is appended (RAG guaranteed)
  * No API key or model "offline-template"        -> templates only (status="success", source="template")
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from .config import OFFLINE_MODEL
from .llm import LLMClient
from .models import FollowUpResult, PatientRecord
from .prompts import build_messages, fmt_date, patient_payload
from .rag import KnowledgeBase, Tip, tip_coverage
from .templates import render_message

logger = logging.getLogger("healthtrack.generator")

TIPS_PER_PATIENT = 2
MAX_MESSAGE_CHARS = 600
TOKENS_PER_PATIENT = 110


@dataclass
class _Work:
    index: int
    patient: PatientRecord
    tips: list[Tip]

    @property
    def ref(self) -> str:
        return str(self.index)


def parse_batch_response(raw: str) -> dict[str, str]:
    """Parse the LLM's JSON into {ref: message}. Raises ValueError if it is not usable."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM output is not valid JSON") from exc
    items = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("LLM JSON has no 'messages' list")
    parsed: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("message"), str) and item.get("ref") is not None:
            parsed[str(item["ref"])] = item["message"]
    return parsed


def mentions_date(message: str, d) -> bool:
    """True if the message names the day and month, e.g. '23 Oct', '23rd October' or 'Oct 23'."""
    months = f"{d.strftime('%b')}|{d.strftime('%B')}"
    pattern = rf"\b{d.day}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{months})\b|\b(?:{months})\s+{d.day}\b"
    return re.search(pattern, message, re.I) is not None


def clean_message(message: str) -> str:
    return " ".join(message.split())[:MAX_MESSAGE_CHARS].strip()


class FollowUpGenerator:
    def __init__(
        self, kb: KnowledgeBase, llm: LLMClient | None = None, *, tip_threshold: float = 0.25, log_message_text: bool = True
    ):
        self.kb = kb
        self.llm = llm
        self.log_message_text = log_message_text
        self.tip_threshold = tip_threshold

    # ------------------------------------------------------------------ public
    async def generate(
        self,
        patients: list[PatientRecord],
        *,
        model: str,
        concurrency: int = 10,
        batch_size: int = 10,
    ) -> list[FollowUpResult]:
        work = [_Work(i, p, self.kb.retrieve(p.condition_key, p.notes, TIPS_PER_PATIENT)) for i, p in enumerate(patients)]
        results: list[FollowUpResult | None] = [None] * len(work)

        if self.llm is None or model == OFFLINE_MODEL:
            for w in work:
                results[w.index] = self._template_result(w, status="success", error=None)
            return self._finish(results)

        semaphore = asyncio.Semaphore(max(1, concurrency))
        batches = [work[i : i + batch_size] for i in range(0, len(work), max(1, batch_size))]

        async def run(batch: list[_Work]) -> None:
            async with semaphore:
                for index, result in await self._run_batch(batch, model):
                    results[index] = result

        await asyncio.gather(*(run(b) for b in batches))
        return self._finish(results)

    # ----------------------------------------------------------------- helpers
    async def _run_batch(self, batch: list[_Work], model: str) -> list[tuple[int, FollowUpResult]]:
        payloads = [patient_payload(w.ref, w.patient, w.tips) for w in batch]
        try:
            raw = await self.llm.complete(  # type: ignore[union-attr]
                model=model,
                messages=build_messages(payloads),
                max_tokens=TOKENS_PER_PATIENT * len(batch) + 64,
            )
            messages = parse_batch_response(raw)
        except Exception as exc:  # noqa: BLE001 - one bad batch must never sink the run
            error = f"{type(exc).__name__}: {exc}"
            logger.error("LLM batch failed, using template fallback | patients=%s | error=%s",
                         [w.patient.patient_id for w in batch], error)
            return [(w.index, self._template_result(w, status="fallback", error=error)) for w in batch]

        out: list[tuple[int, FollowUpResult]] = []
        for w in batch:
            text = clean_message(messages.get(w.ref, ""))
            if not text:
                logger.error("LLM omitted patient, using template fallback | patient=%s", w.patient.patient_id)
                out.append((w.index, self._template_result(w, status="fallback", error="LLM response missed this patient")))
                continue
            out.append((w.index, self._llm_result(w, self._enforce_facts(text, w))))
        return out

    def _ensure_tip(self, message: str, tips: list[Tip]) -> str:
        """Guarantee at least one retrieved health tip is present (paraphrases count)."""
        if any(tip_coverage(message, t) >= self.tip_threshold for t in tips):
            return message
        return f"{message} Tip: {tips[0].text}"

    def _enforce_facts(self, message: str, w: _Work) -> str:
        """Deterministic guard-rails: small models sometimes drop required content."""
        p = w.patient
        message = self._ensure_tip(message, w.tips)
        if p.follow_up_type == "escalation" and not re.search(r"\bcall", message, re.I):
            message += " Our care team will call you soon; please reply or call the clinic."
        elif p.follow_up_type == "reminder" and not re.search(r"repl|respond|let us know|get back|touch", message, re.I):
            message += " Please reply to let us know how you are feeling."
        if not mentions_date(message, p.follow_up_date):
            message += f" Follow-up due by {fmt_date(p.follow_up_date)}."
        return message

    def _base(self, w: _Work) -> dict:
        p = w.patient
        return dict(
            patient_id=p.patient_id, name=p.name, condition=p.condition_label,
            appointment_date=p.appointment_date, follow_up_date=p.follow_up_date,
            follow_up_type=p.follow_up_type, tips=[t.text for t in w.tips],
        )

    def _llm_result(self, w: _Work, message: str) -> FollowUpResult:
        return FollowUpResult(**self._base(w), status="success", source="llm", message=message)

    def _template_result(self, w: _Work, *, status: str, error: str | None) -> FollowUpResult:
        message = render_message(w.patient, [t.text for t in w.tips])
        return FollowUpResult(**self._base(w), status=status, source="template", message=message, error=error)

    def _finish(self, results: list[FollowUpResult | None]) -> list[FollowUpResult]:
        final = [r for r in results if r is not None]
        for r in final:
            # LOG_MESSAGE_TEXT=false keeps message bodies (health information) out of the log files.
            body = f"text={r.message[:240]}" if self.log_message_text else f"chars={len(r.message)}"
            logger.info("message generated | patient=%s | source=%s | status=%s | type=%s | %s",
                        r.patient_id, r.source, r.status, r.follow_up_type, body)
        return final
