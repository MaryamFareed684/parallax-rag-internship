from src.prompts import build_context, build_messages


def test_chunks_are_numbered_with_titles():
    hits = [{"title": "Albedo", "text": "first"}, {"title": "Sun", "text": "second"}]
    context, used = build_context(hits)
    assert context.startswith("[1] (Albedo)")
    assert "[2] (Sun)" in context
    assert len(used) == 2


def test_context_respects_the_size_budget():
    hits = [{"title": "A", "text": "x" * 100}, {"title": "B", "text": "y" * 100}]
    context, used = build_context(hits, max_chars=150)
    assert len(used) == 1
    assert "[2]" not in context


def test_oversized_first_chunk_is_cut():
    context, used = build_context([{"title": "A", "text": "x" * 500}], max_chars=100)
    assert len(used) == 1
    assert len(context) == 100


def test_messages_have_system_rules_and_user_content():
    messages = build_messages("What is albedo?", "[1] (Albedo)\nsome text")
    assert messages[0]["role"] == "system"
    assert "ONLY" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "What is albedo?" in messages[1]["content"]
    assert "some text" in messages[1]["content"]