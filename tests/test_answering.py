from pathlib import Path
from types import SimpleNamespace

from reflective_qa.answering import Answer, RunConfig, build_request, load_runs, parse_message
from reflective_qa.data import load_split

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
RUN = RunConfig(name="test", model="claude-haiku-5-5", k=2, effort="low", max_tokens=100)


def _reply(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           model="claude-haiku-5-5", stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=1200, output_tokens=80))


def test_request_labels_each_passage_with_its_id_and_ends_with_the_question():
    (paper,) = load_split(FIXTURE)
    question = paper.questions[0]
    request = build_request(RUN, paper, question, list(paper.paragraphs[2:4]))
    prompt = request["messages"][0]["content"]
    assert "[s1p0] Experiments. Datasets. We use the Foo dataset." in prompt
    assert "[s1p1]" in prompt and "[s0p0]" not in prompt
    assert prompt.endswith(f"Question: {question.text}")
    assert request["model"] == "claude-haiku-5-5" and request["output_config"]["effort"] == "low"


def test_closed_book_request_has_no_passages_and_asks_for_a_best_guess():
    (paper,) = load_split(FIXTURE)
    prompt = build_request(RUN, paper, paper.questions[0], [])["messages"][0]["content"]
    assert "No passages are given. Give your best guess" in prompt and "[s" not in prompt


def test_reply_parses_to_answer_and_citations_as_written():
    answer = parse_message("q1", _reply('{"answer": "Foo", "evidence": ["s1p0", "s9p9"]}'))
    assert answer == Answer("q1", "claude-haiku-5-5", "Foo", ("s1p0", "s9p9"), "ok", 1200, 80)


def test_refusal_truncation_and_bad_json_give_an_empty_answer_with_the_reason():
    assert parse_message("q", _reply("", "refusal")).status == "refusal"
    assert parse_message("q", _reply('{"answer": "Fo', "max_tokens")).status == "max_tokens"
    bad = parse_message("q", _reply('{"answer": "Foo"}'))
    assert (bad.answer, bad.status, bad.output_tokens) == ("", "invalid_json", 80)


def test_runs_load_from_config():
    runs = load_runs()
    assert {r.k for r in runs.values()} == {20, 0, 5}
