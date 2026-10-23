import json

import src.rag as rag
from src.llm_client import LLMError
from src.prompts import NO_ANSWER

GOOD_ANSWER = "Albedo is the fraction of sunlight reflected by a body. [1]"


def hits(distance=0.2):
    return [{
        "id": "1_0", "title": "Albedo", "chunk_index": 0, "distance": distance,
        "text": "Albedo is the fraction of sunlight that is diffusely reflected by a body.",
    }]


def fake_llm(text):
    def llm(messages):
        return {"text": text, "model": "fake", "prompt_tokens": 10,
                "completion_tokens": 5, "retries": 0, "finish_reason": "stop"}
    return llm


def llm_must_not_be_called(messages):
    raise AssertionError("The LLM should not be called here")


def test_good_question_is_answered(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.2))
    r = rag.answer_question("What is albedo?", llm=fake_llm(GOOD_ANSWER), log=False)
    assert r["status"] == "answered"
    assert r["answer"] == GOOD_ANSWER
    assert r["retrieval_ms"] is not None
    assert r["generation_ms"] is not None
    assert r["total_ms"] is not None


def test_far_results_never_reach_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.9))
    r = rag.answer_question("pizza recipe", llm=llm_must_not_be_called, log=False)
    assert r["status"] == "out_of_domain"
    assert r["generation_ms"] is None


def test_no_results_is_out_of_domain(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: [])
    r = rag.answer_question("anything", llm=llm_must_not_be_called, log=False)
    assert r["status"] == "out_of_domain"


def test_threshold_controls_the_gate(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.45))
    llm = fake_llm(GOOD_ANSWER)
    strict = rag.answer_question("q", threshold=0.40, llm=llm, log=False)
    loose = rag.answer_question("q", threshold=0.50, llm=llm, log=False)
    assert strict["status"] == "out_of_domain"
    assert loose["status"] == "answered"


def test_refusal_from_the_model_is_reported(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.2))
    r = rag.answer_question("q", llm=fake_llm(NO_ANSWER), log=False)
    assert r["status"] == "no_answer"
    assert r["answer"] == NO_ANSWER


def test_unsupported_answer_is_flagged_and_hidden(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.2))
    made_up = "Bananas grow in tropical countries and taste sweet. [1]"
    r = rag.answer_question("q", llm=fake_llm(made_up), log=False)
    assert r["status"] == "flagged"
    assert r["answer"] == rag.UNVERIFIED_MESSAGE
    assert r["raw_answer"] == made_up


def test_api_failure_gives_a_friendly_message(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.2))

    def failing_llm(messages):
        raise LLMError("rate_limit", "too many requests")

    r = rag.answer_question("q", llm=failing_llm, log=False)
    assert r["status"] == "llm_error"
    assert r["error_kind"] == "rate_limit"
    assert "busy" in r["answer"]


def test_empty_question_is_rejected():
    r = rag.answer_question("   ", llm=llm_must_not_be_called, log=False)
    assert r["status"] == "invalid_query"


def test_too_long_prompt_is_retried_with_fewer_chunks(monkeypatch):
    many = hits(0.2) * 4
    monkeypatch.setattr(rag, "search", lambda *a, **k: many)
    calls = []

    def llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise LLMError("context_too_long", "too long")
        return fake_llm(GOOD_ANSWER)(messages)

    r = rag.answer_question("What is albedo?", llm=llm, log=False)
    assert r["status"] == "answered"
    assert len(calls) == 2
    assert len(r["sources"]) < 4


def test_each_question_is_logged_with_latency(monkeypatch, tmp_path):
    log_file = tmp_path / "log.jsonl"
    monkeypatch.setattr(rag, "LOG_PATH", str(log_file))
    monkeypatch.setattr(rag, "search", lambda *a, **k: hits(0.2))
    rag.answer_question("What is albedo?", llm=fake_llm(GOOD_ANSWER), log=True)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "answered"
    assert record["total_ms"] is not None