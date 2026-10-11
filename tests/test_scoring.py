import json
from pathlib import Path

from reflective_qa.data import load_split
from reflective_qa.scoring import (Prediction, answer_f1, evidence_f1, is_unanswerable, load_predictions, normalize,
                                   score, token_f1)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


def _questions():
    return {q.id: q for paper in load_split(FIXTURE) for q in paper.questions}


def test_normalization_ignores_case_punctuation_and_articles():
    assert normalize("The Foo, and a BAR!") == "foo and bar"
    assert token_f1("Foo dataset", "the foo dataset") == 1.0


def test_best_reference_wins_and_its_type_is_reported():
    question = _questions()["q-extractive"]  # references: "Foo, Bar" (extractive) and "Unanswerable" (none)
    assert answer_f1(question, "Unanswerable") == (1.0, "none")
    assert answer_f1(question, "Foo") == (2 / 3, "extractive")


def test_evidence_f1_scores_only_references_that_cite_paragraphs_so_citing_nothing_scores_zero():
    question = _questions()["q-extractive"]  # evidence: ("s1p0",), and () for the unanswerable reference
    assert evidence_f1(question, ("s1p0",)) == evidence_f1(question, ("s1p0", "s1p0")) == 1.0
    assert evidence_f1(question, ()) == evidence_f1(question, ("s0p0",)) == 0.0
    assert evidence_f1(question, ("s1p0", "s0p0")) == 2 / 3


def test_questions_where_no_annotator_cited_a_paragraph_get_no_evidence_score():
    assert evidence_f1(_questions()["q-figure"], ()) is None  # its only evidence is a table


def test_abstentions_are_recognized_after_normalization():
    assert is_unanswerable("Unanswerable") and is_unanswerable(" unanswerable. ") and not is_unanswerable("No")


def test_only_text_only_questions_count_and_missing_ones_score_zero():
    result = score(load_split(FIXTURE), {"q-abstractive": Prediction("a tiny model", ("s0p0",)),
                                         "q-figure": Prediction("90%", ())})
    assert result["questions"] == 3 and result["missing"] == 2  # q-figure is not a text-only question
    assert result["answer_f1"] == result["evidence_f1"] == 1 / 3  # missing predictions score 0 on both
    assert result["evidence_questions"] == 3 and result["predicted_unanswerable"] == 0.0
    abstained = score(load_split(FIXTURE), {"q-boolean": Prediction("unanswerable.", ())})
    assert abstained["predicted_unanswerable"] == 1.0


def test_predictions_load_from_jsonl_with_paragraph_id_evidence(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "predicted_answer": "Yes", "predicted_evidence": ["s0p0"]}) + "\n")
    assert load_predictions(path) == {"q1": Prediction("Yes", ("s0p0",))}
