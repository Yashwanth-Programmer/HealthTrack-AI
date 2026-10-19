"""Deterministic message templates.

Used (a) as the safety net when the LLM fails or omits a patient, and
(b) by the 'offline-template' model so the app runs without any API key.
"""
from __future__ import annotations

from .models import PatientRecord
from .prompts import fmt_date


def render_message(patient: PatientRecord, tips: list[str]) -> str:
    first, doctor = patient.first_name, patient.doctor
    appt, follow_up = fmt_date(patient.appointment_date), fmt_date(patient.follow_up_date)
    tip = tips[0] if tips else ""

    if patient.follow_up_type == "escalation":
        text = (
            f"Hi {first}, we have tried to reach you several times since your {appt} visit with {doctor}. "
            f"Our care team will call you soon; please reply or call the clinic to confirm your follow-up by {follow_up}. "
            f"Tip: {tip}"
        )
    elif patient.follow_up_type == "reminder":
        text = (
            f"Hi {first}, we have not heard back since your {appt} visit with {doctor}. "
            f"Please reply to let us know how you are feeling; your follow-up is due by {follow_up}. "
            f"Tip: {tip}"
        )
    else:
        text = (
            f"Hi {first}, thank you for visiting {doctor} on {appt}. "
            f"Your {patient.condition_label.lower()} follow-up is due by {follow_up}. "
            f"Tip: {tip}"
        )
        if not any(w in tip.lower() for w in ("contact", "call us")):
            text += " Contact us if you feel worse."
    return " ".join(text.split())
