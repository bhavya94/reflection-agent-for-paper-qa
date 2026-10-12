from pathlib import Path
from types import SimpleNamespace

from reflective_qa.agent.writer import ModelAnswer, RunConfig, Writer
from reflective_qa.data.qasper import QasperSplit

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
WRITER = Writer(RunConfig(name="test", model="claude-haiku-5-5", k=2, effort="low", max_tokens=100))


def _reply(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           model="claude-haiku-5-5", stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=1200, output_tokens=80))


def test_request_labels_each_passage_with_its_id_and_ends_with_the_question():
    (paper,) = QasperSplit.load(FIXTURE).papers
    question = paper.questions[0]
    request = WRITER.request(paper, question, list(paper.paragraphs[2:4]))
    prompt = request["messages"][0]["content"]
    assert "[s1p0] Experiments. Datasets. We use the Foo dataset." in prompt
    assert "[s1p1]" in prompt and "[s0p0]" not in prompt
    assert prompt.endswith(f"Question: {question.text}")
    assert request["model"] == "claude-haiku-5-5" and request["output_config"]["effort"] == "low"


def test_closed_book_request_has_no_passages_and_asks_for_a_best_guess():
    (paper,) = QasperSplit.load(FIXTURE).papers
    prompt = WRITER.request(paper, paper.questions[0], [])["messages"][0]["content"]
    assert "No passages are given. Give your best guess" in prompt and "[s" not in prompt


def test_reply_parses_to_answer_and_citations_as_written():
    answer = Writer.parse("q1", _reply('{"answer": "Foo", "evidence": ["s1p0", "s9p9"]}'))
    assert answer == ModelAnswer("q1", "claude-haiku-5-5", "Foo", ("s1p0", "s9p9"), "ok", 1200, 80)


def test_cited_ids_lose_brackets_copied_from_the_passage_labels():
    answer = Writer.parse("q1", _reply('{"answer": "Foo", "evidence": ["[s1p0]", " [s2p3] ", "s4p1"]}'))
    assert answer.evidence_ids == ("s1p0", "s2p3", "s4p1")


def test_refusal_truncation_and_bad_json_give_an_empty_answer_with_the_reason():
    assert Writer.parse("q", _reply("", "refusal")).status == "refusal"
    assert Writer.parse("q", _reply('{"answer": "Fo', "max_tokens")).status == "max_tokens"
    bad = Writer.parse("q", _reply('{"answer": "Foo"}'))
    assert (bad.answer, bad.status, bad.output_tokens) == ("", "invalid_json", 80)


def test_runs_load_from_config():
    runs = Writer.load_all()
    assert {r.k for r in runs.values()} == {20, 0, 5}
