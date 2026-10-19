"""MistralClient against a mocked HTTP transport: no network, no real key."""
import asyncio
import json

import httpx
import pytest

from healthtrack.llm import LLMError, MistralClient


def make_client(handler, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return MistralClient("test-key-123", http=http, **kw)


def ok_response(content='{"messages":[]}'):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def complete(client):
    return asyncio.run(client.complete(model="mistral-small-latest", messages=[{"role": "user", "content": "hi"}], max_tokens=50))


def test_sends_auth_header_json_mode_and_returns_content():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok_response("hello")

    assert complete(make_client(handler)) == "hello"
    assert seen["auth"] == "Bearer test-key-123"
    assert seen["body"]["response_format"] == {"type": "json_object"} and seen["body"]["model"] == "mistral-small-latest"


def test_default_timeout_is_15_seconds():
    client = make_client(lambda r: ok_response())
    assert client._timeout == 15.0


def test_retries_on_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "0"}) if len(calls) == 1 else ok_response("done")

    assert complete(make_client(handler, timeout=4.0)) == "done" and len(calls) == 2


def test_gives_up_within_budget_and_never_leaks_key():
    def handler(request):
        return httpx.Response(503)

    with pytest.raises(LLMError) as exc:
        complete(make_client(handler, timeout=1.0, max_retries=5))
    assert "503" in str(exc.value) and "test-key-123" not in str(exc.value)


def test_non_retryable_error_fails_fast():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(LLMError, match="401"):
        complete(make_client(handler))
    assert len(calls) == 1


def test_malformed_response_raises_llm_error():
    with pytest.raises(LLMError):
        complete(make_client(lambda r: httpx.Response(200, json={"oops": 1})))
