import csv

from conftest import FakeLLM

from healthtrack import cli
from healthtrack.config import BASE_DIR

SAMPLE = BASE_DIR / "data" / "sample_patients_100.csv"


def test_cli_prints_one_message_per_patient_and_writes_csv(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    out = tmp_path / "results.csv"
    code = cli.main([str(SAMPLE), "--model", "offline-template", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and text.count("] ") >= 100 and "100/100 messages" in text
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 100 and all(r["message"] and r["tips"] for r in rows)
    assert (tmp_path / "logs" / "healthtrack.log").exists()


def test_cli_with_llm_client(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    cli.main([str(SAMPLE), "--model", "mistral-small-latest", "--quiet"], llm=FakeLLM())
    assert "AI 100" in capsys.readouterr().out
