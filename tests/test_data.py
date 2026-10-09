from pathlib import Path

import pytest

from reflective_qa.data import AnswerType, Question, load_split, text_only_questions

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


@pytest.fixture(scope="module")
def questions() -> dict[str, Question]:
    papers = load_split(FIXTURE)
    return {q.id: q for paper in papers for q in paper.questions}


def test_empty_paragraphs_are_dropped_and_ids_stay_positional():
    (paper,) = load_split(FIXTURE)
    assert [p.id for p in paper.paragraphs] == ["s0p0", "s0p2", "s1p0"]
    assert paper.paragraphs[2].section == "Experiments ::: Datasets"


def test_answer_text_and_type_match_the_official_evaluator(questions):
    extractive, unanswerable = questions["q-extractive"].answers
    assert (extractive.text, extractive.type) == ("Foo, Bar", AnswerType.EXTRACTIVE)
    assert (unanswerable.text, unanswerable.type) == ("Unanswerable", AnswerType.NONE)
    assert questions["q-abstractive"].answers[0].type == AnswerType.ABSTRACTIVE
    assert questions["q-boolean"].answers[0].text == "No"


def test_evidence_maps_to_paragraphs_despite_whitespace_differences(questions):
    assert questions["q-abstractive"].answers[0].evidence_ids == ("s0p0",)


def test_heading_evidence_is_kept_separately_not_dropped(questions):
    answer = questions["q-extractive"].answers[0]
    assert answer.evidence_ids == ("s1p0",)
    assert answer.other_evidence == ("Datasets",)


def test_text_only_excludes_questions_that_need_a_figure_or_table():
    ids = {q.id for q in text_only_questions(load_split(FIXTURE))}
    assert ids == {"q-extractive", "q-abstractive", "q-boolean"}
