from healthtrack.config import load_settings


def test_invalid_request_timeout_uses_15_second_default(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "not-a-number")
    assert load_settings().request_timeout == 15.0


def test_request_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "24")
    assert load_settings().request_timeout == 24.0
