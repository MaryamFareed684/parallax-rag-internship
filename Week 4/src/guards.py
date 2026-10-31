import re

from src.prompts import NO_ANSWER

def normalize_citations(text):
    """Turn citation styles like 【1】 or [1, 2] into the standard [1][2] form."""
    text = text.replace("【", "[").replace("】", "]")
    text = text.replace("［", "[").replace("］", "]")

    def expand(match):
        return "".join(f"[{n}]" for n in re.findall(r"\d+", match.group(1)))

    return re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]", expand, text)

STOPWORDS = set(
    "the a an and or but of in on at to for with by from as is are was were be been "
    "this that these those it its his her their they he she we you i not no also "
    "which who whom what when where how than then there has have had can could "
    "would should will may might into about over after before between during".split()
)


def is_out_of_domain(hits, threshold):
    """True when nothing was found, or even the best chunk is too far away."""
    return not hits or hits[0]["distance"] > threshold


def _stem(word):
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def check_grounding(answer, context, num_passages, min_support=0.6):
    """Check whether the answer is supported by the context.

    Returns a dict with: flagged (True/False), reasons (list), support (0 to 1).
    """
    answer = normalize_citations(answer)
    if answer.strip() == NO_ANSWER:
        return {"flagged": False, "reasons": [], "support": 1.0}

    reasons = []

    # 1. Citations must point to passages that exist
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        reasons.append("no_citations")
    if any(n < 1 or n > num_passages for n in cited):
        reasons.append("invalid_citation")

    # 2. Numbers in the answer must appear in the context
    clean = re.sub(r"\[\d+\]", " ", answer)
    context_numbers = set(re.findall(r"\d+", context))
    answer_numbers = set(re.findall(r"\d+", clean))
    if answer_numbers - context_numbers:
        reasons.append("number_not_in_context")

    # 3. Most sentences must be made of words that exist in the context
    context_words = _content_words(context)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    supported = 0
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            supported += 1
            continue
        if len(words & context_words) / len(words) >= min_support:
            supported += 1
    support = supported / len(sentences) if sentences else 0.0
    if support < min_support:
        reasons.append("low_word_support")

    return {"flagged": bool(reasons), "reasons": reasons, "support": round(support, 2)}