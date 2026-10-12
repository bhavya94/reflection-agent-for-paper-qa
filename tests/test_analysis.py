from pathlib import Path

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.analysis import Revision, RevisionAnalysis

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
QUESTIONS = {q.id: q for q in QasperSplit.load(FIXTURE).questions()}
DRAFT = "a tiny model with many words"


def test_analysis_splits_changes_by_kind_and_brackets_the_mean_change():
    revisions = [Revision(QUESTIONS["q-boolean"], "Unanswerable", "No"),         # abstention turned into the right answer
                 Revision(QUESTIONS["q-abstractive"], DRAFT, "Unanswerable"),     # a right answer turned into an abstention
                 Revision(QUESTIONS["q-extractive"], DRAFT, "Foo")]               # reworded, and better
    result = RevisionAnalysis(resamples=500, seed=0).analyze(revisions)
    assert {k: v["n"] for k, v in result["kinds"].items()} == {"abstain_to_answer": 1, "answer_to_abstain": 1, "reworded": 1}
    assert result["kinds"]["abstain_to_answer"]["mean_delta"] == 1.0
    assert result["ci"][0] <= result["delta"] <= result["ci"][1]


def test_unchanged_answers_are_not_counted_as_changes():
    result = RevisionAnalysis(resamples=100, seed=0).analyze([Revision(q, DRAFT, DRAFT) for q in QUESTIONS.values()])
    assert result["delta"] == 0.0 and all(v["n"] == 0 for v in result["kinds"].values())
