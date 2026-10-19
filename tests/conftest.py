import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from healthtrack.config import BASE_DIR  # noqa: E402
from healthtrack.llm import LLMError  # noqa: E402
from healthtrack.rag import KnowledgeBase  # noqa: E402

TODAY = date(2026, 10, 18)


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_file(BASE_DIR / "data" / "knowledge_base.json")


def raw_row(**overrides):
    row = {
        "patient_id": "P001", "patient_name": "Rahul Sharma", "condition": "Diabetes",
        "appointment_date": "2026-10-15", "doctor": "Dr. Ramesh", "phone": "9876543210",
        "notes": "", "response_status": "", "attempts": "0",
    }
    row.update(overrides)
    return row


class FakeLLM:
    """Stand-in for Mistral: echoes each patient's tips into a JSON reply."""

    def __init__(self, *, latency=0.0, fail=False, drop_every=0, include_tips=True, garbage=False):
        self.latency, self.fail, self.drop_every = latency, fail, drop_every
        self.include_tips, self.garbage = include_tips, garbage
        self.calls = 0
        self.models = []

    async def complete(self, *, model, messages, max_tokens):
        self.calls += 1
        self.models.append(model)
        await asyncio.sleep(self.latency)
        if self.fail:
            raise LLMError("HTTP 429; gave up within 4.0s budget")
        if self.garbage:
            return "this is not json"
        patients = json.loads(messages[-1]["content"])["patients"]
        out = []
        for i, p in enumerate(patients):
            if self.drop_every and i % self.drop_every == 0:
                continue
            text = f"Hi {p['first_name']}, thanks for visiting {p['doctor']}."
            if self.include_tips:
                text += " " + p["tips"][0]
            out.append({"ref": p["ref"], "message": text})
        return json.dumps({"messages": out})
