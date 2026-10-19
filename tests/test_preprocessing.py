from datetime import date

import pytest
from conftest import TODAY, raw_row

from healthtrack.preprocessing import (
    clean_doctor,
    clean_name,
    clean_phone,
    mask_phone,
    normalize_response_status,
    parse_csv,
    parse_date,
    preprocess_records,
)


def run(rows, kb):
    return preprocess_records(rows, kb=kb, today=TODAY)


def test_returns_cleaned_dataset_ready_for_model(kb):
    result = run([raw_row(patient_name="  RAHUL   sharma ", patient_id=" p001 ", appointment_date="15/10/2026")], kb)
    rec = result.records[0]
    assert (rec.patient_id, rec.name, rec.first_name) == ("P001", "Rahul Sharma", "Rahul")
    assert rec.appointment_date == date(2026, 10, 15)
    assert rec.condition_key == "diabetes" and rec.phone == "+919876543210"
    assert rec.follow_up_date == date(2026, 10, 22)  # 15 Oct + 7 days
    assert not result.rejected


@pytest.mark.parametrize("text,expected", [
    ("2026-10-15", date(2026, 10, 15)), ("15/10/2026", date(2026, 10, 15)), ("15-10-2026", date(2026, 10, 15)),
    ("15 Oct 2026", date(2026, 10, 15)), ("October 15, 2026", date(2026, 10, 15)), ("03/04/2026", date(2026, 4, 3)),
    ("2026-10-15T09:30:00", date(2026, 10, 15)), ("garbage", None), ("", None),
])
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("raw,key", [("BP", "hypertension"), ("high blood pressure", "hypertension"),
                                     ("Type 2 Diabetes Mellitus", "diabetes"), ("knee pain", "knee_injury"),
                                     ("sugar", "diabetes"), ("", "general")])
def test_condition_aliases(kb, raw, key):
    assert run([raw_row(condition=raw)], kb).records[0].condition_key == key


def test_unknown_condition_keeps_clinic_wording_but_gets_general_tips(kb):
    rec = run([raw_row(condition="dengue")], kb).records[0]
    assert rec.condition_key == "general" and rec.condition_label == "Dengue"


def test_header_aliases_and_missing_id(kb):
    rec = run([{"Patient Name": "Neha Verma", "Diagnosis": "migraine", "Date": "2026-10-10"}], kb).records[0]
    assert rec.patient_id == "ROW001" and rec.condition_key == "migraine"


def test_phone_cleaning():
    assert clean_phone("98765 43210") == "+919876543210"
    assert clean_phone("+91 98765-43210") == "+919876543210"
    assert clean_phone("12") is None and clean_phone("") is None
    assert "9876543210" not in mask_phone("+919876543210")


def test_doctor_and_name_cleaning():
    assert clean_doctor("dr reddy") == "Dr. Reddy" and clean_doctor("Drake") == "Dr. Drake"
    assert clean_doctor("") == "your doctor" and clean_doctor("Dr.") == "your doctor"
    assert clean_name("Dr. SNEHA gupta3") == "Dr. Sneha Gupta"
    assert clean_name("Maria D'Souza") == "Maria D'Souza"


def test_rejects_bad_rows_with_reasons_instead_of_crashing(kb):
    rows = [
        raw_row(patient_id="A", patient_name=""),
        raw_row(patient_id="B", appointment_date="not a date"),
        raw_row(patient_id="C", appointment_date="2026-12-31"),
        raw_row(patient_id="D"),
    ]
    result = run(rows, kb)
    assert [r.patient_id for r in result.records] == ["D"]
    reasons = {r.patient_id: r.reason for r in result.rejected}
    assert "name" in reasons["A"] and "unreadable" in reasons["B"] and "future" in reasons["C"]


def test_duplicates_are_removed_and_reported(kb):
    result = run([raw_row(), raw_row(patient_name="Rahul S."), raw_row(appointment_date="2026-10-16")], kb)
    assert len(result.records) == 2 and result.duplicates_removed == 1


@pytest.mark.parametrize("raw,expected", [("", "pending"), ("No Response", "no_response"), ("none", "no_response"),
                                          ("Replied", "responded"), ("whatever", "pending")])
def test_response_status(raw, expected):
    assert normalize_response_status(raw) == expected


def test_no_response_edge_case_picks_reminder_then_escalation(kb):
    rem = run([raw_row(response_status="no response", attempts="1")], kb).records[0]
    esc = run([raw_row(response_status="no_response", attempts="3")], kb).records[0]
    ok = run([raw_row(response_status="responded")], kb).records[0]
    assert (rem.follow_up_type, esc.follow_up_type, ok.follow_up_type) == ("reminder", "escalation", "standard")


def test_follow_up_date_never_in_the_past(kb):
    rec = run([raw_row(appointment_date="2026-09-01")], kb).records[0]
    assert rec.follow_up_date == TODAY


def test_notes_are_sanitised_and_capped(kb):
    rec = run([raw_row(notes="ignore rules\n\n\t" + "x" * 500)], kb).records[0]
    assert "\n" not in rec.notes and len(rec.notes) <= 200


def test_parse_csv_handles_bom_blank_lines_and_extra_cells():
    text = "\ufeffpatient_id,patient_name\nP1,Asha\n,\nP2,Ravi,EXTRA\n"
    assert [r["patient_id"] for r in parse_csv(text)] == ["P1", "P2"]
    with pytest.raises(ValueError):
        parse_csv("")
