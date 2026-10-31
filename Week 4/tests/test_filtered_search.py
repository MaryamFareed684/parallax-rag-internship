import numpy as np
import pytest

from src.filtered_search import build_where, filtered_search, find_topic
from src.vector_store import get_collection, ingest


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.ones((len(texts), 8))


def make_collection(tmp_path):
    col = get_collection(path=str(tmp_path), name="test_col")
    metas = [
        {"title": "A", "topic_id": 1, "has_person": True, "topic_similarity": 0.9},
        {"title": "B", "topic_id": 1, "has_person": False, "topic_similarity": 0.4},
        {"title": "C", "topic_id": 2, "has_person": True, "topic_similarity": 0.8},
        {"title": "D", "topic_id": 2, "has_person": False, "topic_similarity": 0.7},
    ]
    ingest(col, ["a", "b", "c", "d"], ["text a", "text b", "text c", "text d"],
           np.random.rand(4, 8), metas)
    return col


def test_no_options_means_no_filter():
    assert build_where() is None


def test_single_option_is_a_simple_filter():
    assert build_where(topic_id=3) == {"topic_id": 3}
    assert build_where(min_similarity=0.5) == {"topic_similarity": {"$gte": 0.5}}


def test_several_options_are_combined_with_and():
    assert build_where(topic_id=3, has_person=True) == {
        "$and": [{"topic_id": 3}, {"has_person": True}]}


def test_topic_filter_only_returns_that_topic(tmp_path):
    col = make_collection(tmp_path)
    hits = filtered_search("q", k=5, where=build_where(topic_id=1), collection=col, model=FakeModel())
    assert len(hits) == 2
    assert all(h["metadata"]["topic_id"] == 1 for h in hits)


def test_combined_filter(tmp_path):
    col = make_collection(tmp_path)
    where = build_where(topic_id=2, has_person=True)
    hits = filtered_search("q", where=where, collection=col, model=FakeModel())
    assert [h["title"] for h in hits] == ["C"]


def test_similarity_filter(tmp_path):
    col = make_collection(tmp_path)
    hits = filtered_search("q", where=build_where(min_similarity=0.75), collection=col, model=FakeModel())
    assert {h["title"] for h in hits} == {"A", "C"}


def test_filter_with_no_match_returns_empty_list(tmp_path):
    col = make_collection(tmp_path)
    assert filtered_search("q", where=build_where(topic_id=99), collection=col, model=FakeModel()) == []


def test_empty_query_and_bad_k_raise(tmp_path):
    col = make_collection(tmp_path)
    with pytest.raises(ValueError):
        filtered_search("  ", collection=col, model=FakeModel())
    with pytest.raises(ValueError):
        filtered_search("q", k=0, collection=col, model=FakeModel())


def test_find_topic_picks_the_biggest_match(tmp_path):
    path = tmp_path / "topics.csv"
    path.write_text("topic_id,label,docs_total\n1,music band,10\n2,music song,50\n3,film,20\n")
    assert find_topic("music", path=str(path)) == 2
    with pytest.raises(ValueError):
        find_topic("zebra", path=str(path))
