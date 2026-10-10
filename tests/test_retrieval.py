from pathlib import Path

from reflective_qa.data import Answer, AnswerType, Paper, Paragraph, Question, load_split
from reflective_qa.retrieval import PaperRetriever
from reflective_qa.retrieval_eval import EvidenceHit, evaluate, evidence_hit

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


def _answer(*evidence_ids: str) -> Answer:
    return Answer(text="x", type=AnswerType.ABSTRACTIVE, evidence_ids=evidence_ids, other_evidence=(),
                  uses_figure_or_table=False)


def _question(*answers: Answer) -> Question:
    return Question(id="q", paper_id="p", text="?", answers=answers)


def test_paragraph_sharing_the_question_words_ranks_first():
    (paper,) = load_split(FIXTURE)
    assert PaperRetriever(paper).retrieve("Which datasets do they use?", k=1)[0].id == "s1p0"


def test_a_word_only_in_the_section_title_matches_after_stemming():
    # The target's only match is its section title "Dataset", singular. The distractor contains the question's
    # exact word "datasets" in its text, so it would win without section titles or without stemming.
    target = Paragraph(id="target", section="Dataset", text="We use SQuAD and NewsQA.")
    distractor = Paragraph(id="distractor", section="Training", text=(
        "Training runs for ten epochs with a warmup schedule, gradient clipping and mixed precision on all "
        "datasets, and checkpoints are kept every thousand steps."))
    paper = Paper(id="p", title="", abstract="", paragraphs=(distractor, target), questions=())
    assert PaperRetriever(paper).retrieve("Which datasets?", k=1)[0].id == "target"


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


def test_evaluate_scores_each_k_separately_against_a_random_baseline():
    rows = {r["k"]: r for r in evaluate(load_split(FIXTURE), ks=[1, 4], random_samples=2000, seed=0)}
    assert rows[4]["questions"] == 3  # the unanswerable-only and figure questions are not scored
    assert rows[4]["recall"] == rows[4]["all_found"] == rows[4]["random_recall"] == 1.0  # k covers the paper
    assert abs(rows[1]["random_recall"] - 0.25) < 0.03  # one evidence paragraph out of four
    assert rows[1]["median_chars"] < rows[4]["median_chars"]


def test_random_baseline_also_takes_the_best_annotator():
    # Two annotators cite different paragraphs of a two-paragraph paper: any single random pick covers one of them.
    paragraphs = (Paragraph(id="a", section="", text="alpha"), Paragraph(id="b", section="", text="beta"))
    question = Question(id="q", paper_id="p", text="?", answers=(_answer("a"), _answer("b")))
    paper = Paper(id="p", title="", abstract="", paragraphs=paragraphs, questions=(question,))
    (row,) = evaluate([paper], ks=[1], random_samples=50, seed=0)
    assert row["random_recall"] == row["random_all_found"] == 1.0


def test_recall_at_k_counts_only_the_top_k():
    paragraphs = (Paragraph(id="a", section="", text="alpha"), Paragraph(id="b", section="", text="beta"))
    question = Question(id="q", paper_id="p", text="alpha", answers=(_answer("a", "b"),))
    paper = Paper(id="p", title="", abstract="", paragraphs=paragraphs, questions=(question,))
    rows = {r["k"]: r for r in evaluate([paper], ks=[1, 2], random_samples=1, seed=0)}
    assert (rows[1]["recall"], rows[1]["all_found"]) == (0.5, False)
    assert (rows[2]["recall"], rows[2]["all_found"]) == (1.0, True)
