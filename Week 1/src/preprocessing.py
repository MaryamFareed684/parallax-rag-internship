import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


def strip_html(text: str) -> str:
    """Remove HTML tags like <p> and <b>."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """Turn odd characters into standard ones (for example the ligature 'ﬁ' becomes 'fi')."""
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    """Turn many spaces, tabs and new lines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """Return True if the text is English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """Run all cleaning steps in order."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    return text