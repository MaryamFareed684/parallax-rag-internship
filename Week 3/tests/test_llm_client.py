from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm_client import LLMError, generate, generate_with_fallback

MESSAGES = [{"role": "user", "content": "hi"}]


def make_error(cls, status, message="error"):
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


def good_response(text="hello"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def test_success_returns_text_and_token_counts():
    client = FakeClient([good_response("hello")])
    result = generate(MESSAGES, client=client, model="m")
    assert result["text"] == "hello"
    assert result["prompt_tokens"] == 10
    assert result["retries"] == 0


def test_rate_limit_is_retried():
    sleeps = []
    client = FakeClient([make_error(openai.RateLimitError, 429), good_response()])
    result = generate(MESSAGES, client=client, model="m", sleep=sleeps.append)
    assert result["retries"] == 1
    assert len(sleeps) == 1


def test_gives_up_after_max_retries():
    client = FakeClient([make_error(openai.RateLimitError, 429)] * 4)
    with pytest.raises(LLMError) as exc:
        generate(MESSAGES, client=client, model="m", sleep=lambda s: None)
    assert exc.value.kind == "rate_limit"
    assert client.completions.calls == 4


def test_timeout_is_retried():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://test"))
    client = FakeClient([timeout, good_response()])
    result = generate(MESSAGES, client=client, model="m", sleep=lambda s: None)
    assert result["retries"] == 1


def test_server_error_is_retried():
    client = FakeClient([make_error(openai.InternalServerError, 500), good_response()])
    result = generate(MESSAGES, client=client, model="m", sleep=lambda s: None)
    assert result["retries"] == 1


def test_wrong_key_is_not_retried():
    client = FakeClient([make_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError) as exc:
        generate(MESSAGES, client=client, model="m", sleep=lambda s: None)
    assert exc.value.kind == "auth"
    assert client.completions.calls == 1


def test_unknown_model_gives_clear_error():
    client = FakeClient([make_error(openai.NotFoundError, 404)])
    with pytest.raises(LLMError) as exc:
        generate(MESSAGES, client=client, model="m")
    assert exc.value.kind == "model_not_found"


def test_prompt_too_long_is_reported():
    error = make_error(openai.BadRequestError, 400, "maximum context length exceeded")
    client = FakeClient([error])
    with pytest.raises(LLMError) as exc:
        generate(MESSAGES, client=client, model="m")
    assert exc.value.kind == "context_too_long"


def test_missing_model_setting_is_reported(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMError) as exc:
        generate(MESSAGES, client=FakeClient([]), model=None)
    assert exc.value.kind == "config"


def test_backup_model_is_used_when_first_is_missing():
    client = FakeClient([make_error(openai.NotFoundError, 404), good_response("from backup")])
    result = generate_with_fallback(MESSAGES, models=["a", "b"], client=client, sleep=lambda s: None)
    assert result["model"] == "b"
    assert result["text"] == "from backup"


def test_wrong_key_does_not_switch_models():
    client = FakeClient([make_error(openai.AuthenticationError, 401), good_response()])
    with pytest.raises(LLMError) as exc:
        generate_with_fallback(MESSAGES, models=["a", "b"], client=client, sleep=lambda s: None)
    assert exc.value.kind == "auth"
    assert client.completions.calls == 1