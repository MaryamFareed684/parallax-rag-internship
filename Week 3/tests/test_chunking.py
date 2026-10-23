import pytest
from src.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_max_size():
    text = "This is a sentence about science. " * 100
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_shared_between_neighbours():
    text = "".join(f"Sentence number {i} is here. " for i in range(50))
    chunks = chunk_text(text, chunk_size=150, overlap=40)
    for a, b in zip(chunks, chunks[1:]):
        assert b[:10] in a


def test_no_words_are_lost():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert set(text.split()) <= set(" ".join(chunks).split())


def test_bad_settings_raise_error():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_very_long_word_is_still_split():
    chunks = chunk_text("a" * 1000, chunk_size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)