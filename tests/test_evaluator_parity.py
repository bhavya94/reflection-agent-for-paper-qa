"""On the real dev split, the loader must parse answers exactly as the official evaluator does."""

import importlib.util
import json
from pathlib import Path

import pytest

from reflective_qa.data import load_split

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DEV = RAW / "qasper-dev-v0.3.json"
EVALUATOR = RAW / "qasper_evaluator.py"

pytestmark = pytest.mark.skipif(
    not (DEV.exists() and EVALUATOR.exists()), reason="run `python -m reflective_qa.download` first"
)


def test_answer_text_and_type_match_the_official_evaluator_on_dev():
    spec = importlib.util.spec_from_file_location("qasper_evaluator", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    official = evaluator.get_answers_and_evidence(json.loads(DEV.read_text()), text_evidence_only=True)
    ours = {q.id: [(a.text, a.type.value) for a in q.answers] for paper in load_split(DEV) for q in paper.questions}
    assert ours == {qid: [(r["answer"], r["type"]) for r in refs] for qid, refs in official.items()}
