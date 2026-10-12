"""On the real dev split, the loader must parse answers exactly as the official evaluator does."""

import importlib.util
import json
from pathlib import Path

import pytest

from reflective_qa.data.qasper import QasperSplit

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DEV = RAW / "qasper-dev-v0.3.json"
EVALUATOR = RAW / "qasper_evaluator.py"

pytestmark = pytest.mark.skipif(
    not (DEV.exists() and EVALUATOR.exists()), reason="run `uv run rqa download` first"
)


def test_answer_text_and_type_match_the_official_evaluator_on_dev():
    spec = importlib.util.spec_from_file_location("qasper_evaluator", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    official = evaluator.get_answers_and_evidence(json.loads(DEV.read_text()), text_evidence_only=True)
    ours = {q.id: [(a.text, a.type.value) for a in q.answers] for q in QasperSplit.load(DEV).questions()}
    assert ours == {qid: [(r["answer"], r["type"]) for r in refs] for qid, refs in official.items()}


def test_answer_f1_matches_the_official_evaluator_on_dev():
    from reflective_qa.scoring import Prediction, score

    spec = importlib.util.spec_from_file_location("qasper_evaluator", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    split = QasperSplit.load(DEV)
    questions = split.text_only_questions()
    # A mix of exact, partial, wrong, unanswerable and missing predictions; every 7th question has none.
    styles = [lambda q: q.answers[0].text, lambda q: q.answers[-1].text.split()[0], lambda q: "the proposed model",
              lambda q: "Unanswerable", lambda q: "Yes", lambda q: "No", lambda q: "", lambda q: " unanswerable. ",
              lambda q: q.answers[-1].text.upper() + "!!"]
    predictions = {q.id: Prediction(styles[i % len(styles)](q), ()) for i, q in enumerate(questions) if i % 7}
    gold = evaluator.get_answers_and_evidence(json.loads(DEV.read_text()), text_evidence_only=True)
    official = evaluator.evaluate({q.id: gold[q.id] for q in questions},
                                  {qid: {"answer": p.answer, "evidence": []} for qid, p in predictions.items()})
    ours = score(split, predictions)
    assert abs(ours["answer_f1"] - official["Answer F1"]) < 1e-12
    assert set(ours["answer_f1_by_type"]) == set(official["Answer F1 by type"])  # all four types present
    assert ours["answer_f1_by_type"] == pytest.approx(official["Answer F1 by type"], abs=1e-12)
    assert ours["missing"] == official["Missing predictions"]
