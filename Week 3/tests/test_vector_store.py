import numpy as np
import pytest

from src.vector_store import get_collection, ingest
from src.search import search


def test_ingest_skips_empty_and_duplicate_chunks(tmp_path):
    col = get_collection(path=str(tmp_path), name="test")
    stored = ingest(
        col,
        ["a", "b", "a", "c"],
        ["text a", "   ", "text a again", "text c"],
        np.random.rand(4, 8),
        [{"title": "t"}] * 4,
    )
    assert stored == 2
    assert col.count() == 2


def test_ingest_twice_does_not_duplicate(tmp_path):
    col = get_collection(path=str(tmp_path), name="test")
    args = (["a", "b"], ["text a", "text b"], np.random.rand(2, 8), [{"title": "t"}] * 2)
    ingest(col, *args)
    ingest(col, *args)
    assert col.count() == 2


def test_ingest_length_mismatch_raises(tmp_path):
    col = get_collection(path=str(tmp_path), name="test")
    with pytest.raises(ValueError):
        ingest(col, ["a", "b"], ["x"], np.random.rand(2, 8), [{"title": "t"}] * 2)


def test_search_empty_query_raises():
    with pytest.raises(ValueError):
        search("   ")


def test_search_bad_k_raises():
    with pytest.raises(ValueError):
        search("hello", k=0)


def test_search_empty_collection_returns_empty_list(tmp_path):
    col = get_collection(path=str(tmp_path), name="empty")
    assert search("hello", collection=col) == []