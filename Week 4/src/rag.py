import json
import os
import time
from datetime import datetime

from src.guards import check_grounding, is_out_of_domain, normalize_citations
from src.llm_client import LLMError, generate_with_fallback
from src.prompts import NO_ANSWER, build_context, build_messages
from src.search import search

DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.55"))
TOP_K = 5
LOG_PATH = "results/rag_log.jsonl"

OUT_OF_DOMAIN_MESSAGE = (
    "This question seems to be outside the topics in my documents, "
    "so I can't answer it reliably."
)
UNVERIFIED_MESSAGE = (
    "I generated an answer, but I could not verify it against my documents, "
    "so I am not showing it."
)
ERROR_MESSAGES = {
    "auth": "The AI service rejected the API key. Please check the settings.",
    "rate_limit": "The AI service is busy (rate limit). Please try again in a minute.",
    "timeout": "The AI service took too long to answer. Please try again.",
    "connection": "I could not reach the AI service. Please check the internet connection.",
    "model_not_found": "The chosen AI model is not available. Please check the settings.",
}


def _ms(seconds):
    return round(seconds * 1000, 1)


def _write_log(result):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def answer_question(question, k=TOP_K, threshold=None, collection=None,
                    model=None, llm=generate_with_fallback, log=True):
    """Full RAG: retrieve -> gate -> generate -> verify. Always returns a dict."""
    threshold = DISTANCE_THRESHOLD if threshold is None else threshold
    t_start = time.perf_counter()
    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "status": None,
        "answer": None,
        "sources": [],
        "top_distance": None,
        "retrieval_ms": None,
        "generation_ms": None,
        "total_ms": None,
    }

    def finish(status, answer):
        result["status"] = status
        result["answer"] = answer
        result["total_ms"] = _ms(time.perf_counter() - t_start)
        if log:
            _write_log(result)
        return result

    # 1. Retrieve
    t0 = time.perf_counter()
    try:
        hits = search(question, k=k, collection=collection, model=model)
    except ValueError:
        return finish("invalid_query", "Please type a question.")
    result["retrieval_ms"] = _ms(time.perf_counter() - t0)
    result["top_distance"] = round(hits[0]["distance"], 3) if hits else None

    # 2. Gate: out-of-domain questions never reach the LLM
    if is_out_of_domain(hits, threshold):
        return finish("out_of_domain", OUT_OF_DOMAIN_MESSAGE)

    # 3. Generate (if the prompt is too long, retry with fewer chunks)
    used = hits
    t0 = time.perf_counter()
    while True:
        context, used = build_context(used)
        messages = build_messages(question, context)
        try:
            generation = llm(messages)
            break
        except LLMError as e:
            if e.kind == "context_too_long" and len(used) > 1:
                used = used[: max(1, len(used) // 2)]
                continue
            result["generation_ms"] = _ms(time.perf_counter() - t0)
            result["error_kind"] = e.kind
            return finish("llm_error", ERROR_MESSAGES.get(e.kind, "Something went wrong with the AI service."))
    result["generation_ms"] = _ms(time.perf_counter() - t0)

    result["model"] = generation.get("model")
    result["prompt_tokens"] = generation.get("prompt_tokens")
    result["completion_tokens"] = generation.get("completion_tokens")
    result["retries"] = generation.get("retries")
    result["finish_reason"] = generation.get("finish_reason")
    result["sources"] = [{"n": i + 1, "title": h["title"], "distance": round(h["distance"], 3)}
                         for i, h in enumerate(used)]

    text = normalize_citations(generation["text"])
    if not text:
        result["error_kind"] = "empty_response"
        return finish("llm_error", "The AI returned an empty answer. Please try again.")

    # 4. Verify the answer against the chunks
    if text == NO_ANSWER:
        return finish("no_answer", NO_ANSWER)

    check = check_grounding(text, context, num_passages=len(used))
    result["grounding"] = check
    if check["flagged"]:
        result["raw_answer"] = text
        return finish("flagged", UNVERIFIED_MESSAGE)
    return finish("answered", text)