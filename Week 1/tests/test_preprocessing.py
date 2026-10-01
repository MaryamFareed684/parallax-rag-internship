from src.preprocessing import (
    strip_html,
    normalize_unicode,
    clean_whitespace,
    is_english,
    clean_text,
)


def test_strip_html_removes_tags():
    result = strip_html("<p>Hello <b>world</b></p>")
    assert "<" not in result and ">" not in result
    assert "Hello" in result and "world" in result


def test_normalize_unicode_fixes_ligature():
    assert normalize_unicode("ﬁne") == "fine"


def test_normalize_unicode_fixes_fullwidth():
    assert normalize_unicode("ＡＢＣ") == "ABC"


def test_clean_whitespace_collapses_spaces():
    assert clean_whitespace("  a   b \n c ") == "a b c"


def test_clean_whitespace_empty_string():
    assert clean_whitespace("   ") == ""


def test_is_english_true_for_english():
    assert is_english("This is a simple English sentence about machine learning.")


def test_is_english_false_for_french():
    assert not is_english("Ceci est une phrase écrite en français pour le test.")


def test_is_english_false_for_empty():
    assert not is_english("")


def test_clean_text_full_pipeline():
    assert clean_text("<p>Hello    ﬁne\n\n world</p>") == "Hello fine world"