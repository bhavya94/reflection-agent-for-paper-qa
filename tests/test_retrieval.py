from pathlib import Path

from reflective_qa.data import Answer, AnswerType, Question, load_split
from reflective_qa.retrieval import PaperRetriever
from reflective_qa.retrieval_eval import EvidenceHit, evidence_hit

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


def _answer(*evidence_ids: str) -> Answer:
    return Answer(text="x", type=AnswerType.ABSTRACTIVE, evidence_ids=evidence_ids, other_evidence=(),
                  uses_figure_or_table=False)


def _question(*answers: Answer) -> Question:
    return Question(id="q", paper_id="p", text="?", answers=answers)


def test_paragraph_sharing_the_question_words_ranks_first():
    (paper,) = load_split(FIXTURE)
    assert PaperRetriever(paper).retrieve("Which datasets do they use?", k=1)[0].id == "s1p0"


def test_k_larger_than_the_paper_returns_every_paragraph_once():
    (paper,) = load_split(FIXTURE)
    retrieved = PaperRetriever(paper).retrieve("anything", k=50)
    assert sorted(p.id for p in retrieved) == sorted(p.id for p in paper.paragraphs)


def test_recall_takes_the_best_annotator():
    question = _question(_answer("s0p0", "s0p1"), _answer("s2p0"))
    assert evidence_hit(question, {"s2p0"}) == EvidenceHit(recall=1.0, all_found=True)
    assert evidence_hit(question, {"s0p0", "s9p9"}) == EvidenceHit(recall=0.5, all_found=False)


def test_questions_without_paragraph_evidence_are_not_scored():
    assert evidence_hit(_question(_answer()), {"s0p0"}) is None
