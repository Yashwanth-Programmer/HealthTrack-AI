"""Deterministic sample data generator (intentionally a little messy, like real exports)."""
from __future__ import annotations

import csv
import io
import random
from datetime import date, timedelta

FIRST = ["Rahul", "Priya", "Arjun", "Sneha", "Vikram", "Ananya", "Karan", "Neha", "Rohan", "Kavya",
         "Aditya", "Meera", "Sanjay", "Divya", "Imran", "Lakshmi", "Suresh", "Pooja", "Nikhil", "Fatima"]
LAST = ["Sharma", "Mehta", "Patel", "Gupta", "Chen", "Rao", "Singh", "Verma", "Reddy", "Nair",
        "Iyer", "Khan", "Das", "Joshi", "Kulkarni"]
DOCTORS = ["Dr. Ramesh", "Dr. Anita Desai", "Dr. K. Reddy", "Dr. Suresh Menon", "Dr. Farah Ali"]
CONDITIONS = [
    "Diabetes", "type 2 diabetes", "Hypertension", "BP", "Asthma", "High Cholesterol", "Thyroid",
    "Back Pain", "Knee Injury", "Migraine", "Viral fever", "Post-surgery", "Arthritis", "Anemia",
    "acidity", "skin allergy", "UTI", "Pregnancy",
]
NOTES = ["", "", "", "missed evening dose", "reports mild dizziness", "wants diet advice",
         "recovering well", "pain worse at night", "needs repeat lab test", "travelling next week"]
COLUMNS = ["patient_id", "patient_name", "age", "phone", "condition", "appointment_date",
           "doctor", "notes", "response_status", "attempts"]


def generate_sample_rows(n: int = 100, *, seed: int = 42, today: date | None = None) -> list[dict[str, str]]:
    rng = random.Random(seed)
    today = today or date.today()
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    rows = []
    for i in range(n):
        appt = today - timedelta(days=rng.randint(1, 10))
        date_text = rng.choice([
            appt.isoformat(), appt.strftime("%d/%m/%Y"), appt.strftime("%d %b %Y"), appt.strftime("%d-%m-%Y"),
        ])
        name = names[i % len(names)]
        if rng.random() < 0.15:
            name = name.upper()
        elif rng.random() < 0.1:
            name = f"  {name.lower()}  "
        digits = "".join(rng.choice("6789") if j == 0 else rng.choice("0123456789") for j in range(10))
        phone = rng.choice([digits, f"+91 {digits[:5]} {digits[5:]}", f"91{digits}", ""] )
        status = rng.choices(["", "responded", "no_response"], weights=[60, 25, 15])[0]
        rows.append({
            "patient_id": f"P{i + 1:03d}",
            "patient_name": name,
            "age": str(rng.randint(18, 82)),
            "phone": phone,
            "condition": rng.choice(CONDITIONS),
            "appointment_date": date_text,
            "doctor": rng.choice(DOCTORS),
            "notes": rng.choice(NOTES),
            "response_status": status,
            "attempts": str(rng.randint(1, 3)) if status == "no_response" else "0",
        })
    return rows


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
