from dataclasses import replace

import pytest
from conftest import FakeLLM, raw_row
from fastapi.testclient import TestClient

from healthtrack.api import create_app
from healthtrack.config import load_settings


@pytest.fixture
def make_client(tmp_path):
    def _make(llm=None, **overrides):
        settings = replace(load_settings(), api_key=None, log_dir=tmp_path, **overrides)
        return TestClient(create_app(llm=llm, settings=settings))
    return _make


def test_ui_is_served(make_client):
    with make_client() as c:
        r = c.get("/")
        assert r.status_code == 200 and "HealthTrack" in r.text
        assert 'id="fastPreset"' in r.text and "the concurrency setting is a maximum" in r.text
        assert r.headers["x-content-type-options"] == "nosniff"


def test_config_without_key_offers_offline_only(make_client):
    with make_client() as c:
        cfg = c.get("/api/config").json()
        assert cfg["llm_configured"] is False and cfg["models"] == ["offline-template"]


def test_sample_endpoint_gives_100_records_and_csv(make_client):
    with make_client() as c:
        assert len(c.get("/api/sample").json()["records"]) == 100
        csv = c.get("/api/sample.csv")
        assert csv.headers["content-type"].startswith("text/csv") and csv.text.count("\n") == 101


def test_generate_from_records_with_fake_llm(make_client):
    with make_client(llm=FakeLLM()) as c:
        rows = c.get("/api/sample").json()["records"]
        body = c.post("/api/generate", json={"records": rows, "model": "mistral-small-latest"}).json()
        assert body["stats"]["processed"] == 100 and body["stats"]["ai_generated"] == 100
        assert all(r["tips"] and r["message"] for r in body["results"])


def test_generate_from_csv_text_reports_rejections(make_client):
    csv_text = "patient_id,patient_name,condition,appointment_date\nP1,Asha Rao,asthma,2020-01-02\nP2,,asthma,2020-01-02\n"
    with make_client() as c:
        body = c.post("/api/generate", json={"csv_text": csv_text, "model": "offline-template"}).json()
        assert body["stats"]["processed"] == 1 and body["stats"]["rejected"] == 1
        assert "name" in body["rejected"][0]["reason"]


def test_llm_outage_still_returns_messages(make_client):
    with make_client(llm=FakeLLM(fail=True)) as c:
        body = c.post("/api/generate", json={"records": [raw_row()], "model": "mistral-small-latest"}).json()
        assert body["stats"]["fallback"] == 1 and body["results"][0]["message"]


@pytest.mark.parametrize("payload,status", [
    ({}, 422),                                                    # no source
    ({"records": [], "csv_text": "a"}, 422),                      # both sources
    ({"records": []}, 422),                                       # empty
    ({"records": [raw_row()], "model": "gpt-9"}, 422),            # unknown model
    ({"records": [raw_row()], "model": "mistral-small-latest"}, 400),  # needs a key
    ({"records": [raw_row()], "concurrency": 999}, 422),
    ({"csv_text": ""}, 422),
])
def test_input_validation(make_client, payload, status):
    with make_client() as c:
        assert c.post("/api/generate", json=payload).status_code == status


def test_record_and_size_limits(make_client):
    with make_client(max_records=3, max_csv_bytes=1000) as c:
        assert c.post("/api/generate", json={"records": [raw_row()] * 4, "model": "offline-template"}).status_code == 413
        assert c.post("/api/generate", json={"csv_text": "a,b\n" + "x,y\n" * 400}).status_code == 413


def test_fallback_reasons_reported_in_stats(make_client):
    with make_client(llm=FakeLLM(fail=True)) as c:
        rows = [raw_row(patient_id=f"P{i}") for i in range(3)]
        stats = c.post("/api/generate", json={"records": rows, "model": "mistral-small-latest"}).json()["stats"]
        assert stats["fallback"] == 3 and any("429" in r for r in stats["fallback_reasons"])


def test_stats_expose_retry_counter(make_client):
    with make_client(llm=FakeLLM()) as c:
        stats = c.post("/api/generate", json={"records": [raw_row()], "model": "mistral-small-latest"}).json()["stats"]
        assert stats["llm_retries"] == 0


def test_health_and_readiness_probes(make_client):
    with make_client() as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        ready = c.get("/readyz").json()
        assert ready["status"] == "ready" and ready["knowledge_base_conditions"] >= 10


def test_request_id_echoed_and_generated_and_sanitised(make_client):
    with make_client() as c:
        assert c.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
        generated = c.get("/healthz").headers["x-request-id"]
        assert len(generated) == 16
        bad = c.get("/healthz", headers={"X-Request-ID": "bad id\nwith spaces"}).headers["x-request-id"]
        assert bad != "bad id\nwith spaces" and len(bad) == 16


def test_security_headers_present(make_client):
    with make_client() as c:
        h = c.get("/").headers
        assert "default-src 'self'" in h["content-security-policy"]
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
        assert c.get("/api/config").headers["cache-control"] == "no-store"
        assert "content-security-policy" not in c.get("/docs").headers  # swagger needs its CDN


def test_rate_limit_returns_429_with_retry_after(make_client):
    with make_client(rate_limit_per_minute=2) as c:
        body = {"records": [raw_row()], "model": "offline-template"}
        assert [c.post("/api/generate", json=body).status_code for _ in range(3)] == [200, 200, 429]
        limited = c.post("/api/generate", json=body)
        assert int(limited.headers["retry-after"]) >= 1
        assert c.get("/healthz").status_code == 200  # only the expensive endpoint is limited


def test_unhandled_errors_return_clean_500_without_stack_trace(tmp_path):
    from dataclasses import replace

    class Boom:
        async def complete(self, **kwargs):
            raise RuntimeError("secret internal detail")

    app = create_app(llm=Boom(), settings=replace(load_settings(), api_key=None, log_dir=tmp_path))

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
        assert r.status_code == 500 and "secret" not in r.text and r.json()["request_id"]
