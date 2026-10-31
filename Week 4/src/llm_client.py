import os
import random
import time

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

load_dotenv()

FALLBACK_KINDS = {"model_not_found", "rate_limit", "server", "timeout", "connection"}


class LLMError(Exception):
    """The LLM call failed for good. `kind` tells you why."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def make_client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise LLMError("config", "LLM_API_KEY is missing. Add it to your .env file.")
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=0,  # we do the retrying ourselves, so we can count and log it
    )


def _wait_seconds(error, attempt):
    """How long to wait before the next try."""
    try:
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            return min(float(retry_after), 30)
    except Exception:
        pass
    return min(2 ** attempt + random.random(), 20)


def _looks_too_long(error):
    text = str(error).lower()
    return "context" in text or "too long" in text or "token" in text


def generate(messages, client=None, model=None, max_tokens=400,
             temperature=0.0, max_retries=3, sleep=time.sleep):
    """Call the LLM once (with retries). Returns a dict with the text and token counts."""
    client = client or make_client()
    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise LLMError("config", "LLM_MODEL is missing. Add it to your .env file.")

    retries = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = response.choices[0]
            usage = response.usage
            return {
                "text": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "retries": retries,
            }
        except AuthenticationError as e:
            raise LLMError("auth", "The API key was rejected. Check LLM_API_KEY and LLM_BASE_URL.") from e
        except NotFoundError as e:
            raise LLMError("model_not_found", f"Model '{model}' was not found. Check LLM_MODEL.") from e
        except BadRequestError as e:
            kind = "context_too_long" if _looks_too_long(e) else "bad_request"
            raise LLMError(kind, f"The request was rejected: {e}") from e
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if isinstance(e, RateLimitError):
                kind = "rate_limit"
            elif isinstance(e, APITimeoutError):
                kind = "timeout"
            else:
                kind = "connection"
            if retries >= max_retries:
                raise LLMError(kind, f"Failed after {retries} retries ({kind}).") from e
            sleep(_wait_seconds(e, retries))
            retries += 1
        except APIStatusError as e:
            if e.status_code >= 500 and retries < max_retries:
                sleep(_wait_seconds(e, retries))
                retries += 1
                continue
            kind = "server" if e.status_code >= 500 else "unknown"
            raise LLMError(kind, f"API error {e.status_code}.") from e


def generate_with_fallback(messages, models=None, **kwargs):
    """Try the main model, then the backup models from LLM_FALLBACK_MODELS."""
    if models is None:
        backups = os.getenv("LLM_FALLBACK_MODELS", "")
        models = [os.getenv("LLM_MODEL")] + [m.strip() for m in backups.split(",") if m.strip()]
    last_error = None
    for model in models:
        try:
            result = generate(messages, model=model, **kwargs)
            result["model"] = model
            return result
        except LLMError as e:
            last_error = e
            if e.kind not in FALLBACK_KINDS:
                raise
    raise last_error