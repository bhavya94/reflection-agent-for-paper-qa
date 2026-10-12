from pathlib import Path

import pytest

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.records import AnswerType, Question

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


@pytest.fixture(scope="module")
def questions() -> dict[str, Question]:
    return {q.id: q for q in QasperSplit.load(FIXTURE).questions()}


def test_empty_and_repeated_paragraphs_are_dropped_and_ids_stay_positional():
    (paper,) = QasperSplit.load(FIXTURE).papers
    assert [p.id for p in paper.paragraphs] == ["s0p0", "s0p2", "s1p0", "s1p1"]
    assert paper.paragraphs[2].section == "Experiments ::: Datasets"


def test_answer_text_and_type_match_the_official_evaluator(questions):
    extractive, unanswerable = questions["q-extractive"].answers
    assert (extractive.text, extractive.type) == ("Foo, Bar", AnswerType.EXTRACTIVE)
    assert (unanswerable.text, unanswerable.type) == ("Unanswerable", AnswerType.NONE)
    assert questions["q-abstractive"].answers[0].type == AnswerType.ABSTRACTIVE
    assert questions["q-boolean"].answers[0].text == "No"


def test_evidence_maps_to_paragraphs_despite_whitespace_differences(questions):
    assert questions["q-abstractive"].answers[0].evidence_ids == ("s0p0",)


def test_evidence_matching_a_repeated_paragraph_maps_to_its_first_copy(questions):
    assert questions["q-boolean"].answers[0].evidence_ids == ("s0p2",)


def test_heading_evidence_is_kept_separately_even_when_a_paragraph_contains_it(questions):
    answer = questions["q-extractive"].answers[0]
    assert answer.evidence_ids == ("s1p0",)
    assert answer.other_evidence == ("Datasets",)


def test_text_only_excludes_questions_that_need_a_figure_or_table():
    ids = {q.id for q in QasperSplit.load(FIXTURE).text_only_questions()}
    assert ids == {"q-extractive", "q-abstractive", "q-boolean"}


def test_samples_are_seeded_text_only_and_smaller_samples_are_prefixes_of_larger_ones():
    split = QasperSplit.load(FIXTURE)
    assert split.sample(2, seed=0) == split.sample(3, seed=0)[:2]
    assert sorted(q.id for q in split.sample(None, seed=0)) == ["q-abstractive", "q-boolean", "q-extractive"]
    assert split.paper("2101.00001").questions[0].paper_id == "2101.00001"
