from src.guards import check_grounding, is_out_of_domain, normalize_citations
from src.prompts import NO_ANSWER

CONTEXT = (
    "[1] (Albedo)\nAlbedo is the fraction of sunlight that is diffusely reflected "
    "by a body. It is measured on a scale from 0 to 1."
)
GOOD = "Albedo is the fraction of sunlight reflected by a body [1]."


def test_wide_brackets_are_normalized():
    assert normalize_citations("It reflects light【1】.") == "It reflects light[1]."


def test_citation_lists_are_expanded():
    assert normalize_citations("Fact [1, 2].") == "Fact [1][2]."


def test_gate_blocks_when_nothing_found():
    assert is_out_of_domain([], threshold=0.5)


def test_gate_blocks_far_results():
    assert is_out_of_domain([{"distance": 0.8}], threshold=0.5)


def test_gate_allows_close_results():
    assert not is_out_of_domain([{"distance": 0.2}], threshold=0.5)


def test_grounded_answer_passes():
    result = check_grounding(GOOD, CONTEXT, num_passages=1)
    assert not result["flagged"]
    assert result["support"] == 1.0


def test_wide_bracket_citation_is_accepted():
    answer = "Albedo is the fraction of sunlight reflected by a body【1】."
    assert not check_grounding(answer, CONTEXT, num_passages=1)["flagged"]


def test_missing_citation_is_flagged():
    answer = "Albedo is the fraction of sunlight reflected by a body."
    assert "no_citations" in check_grounding(answer, CONTEXT, num_passages=1)["reasons"]


def test_citation_to_missing_passage_is_flagged():
    answer = "Albedo is the fraction of sunlight reflected by a body [5]."
    assert "invalid_citation" in check_grounding(answer, CONTEXT, num_passages=1)["reasons"]


def test_invented_number_is_flagged():
    answer = "Albedo was first measured in 1850 [1]."
    assert "number_not_in_context" in check_grounding(answer, CONTEXT, num_passages=1)["reasons"]


def test_unrelated_answer_is_flagged():
    answer = "Bananas grow in tropical countries and taste sweet [1]."
    assert "low_word_support" in check_grounding(answer, CONTEXT, num_passages=1)["reasons"]


def test_refusal_sentence_is_not_flagged():
    assert not check_grounding(NO_ANSWER, CONTEXT, num_passages=1)["flagged"]