import json
from pathlib import Path
from types import SimpleNamespace

from reflective_qa.checks import Finding
from reflective_qa.critic import CHECKLIST, SCHEMA, SYSTEM, CriticConfig, Critique, build_critic_request, parse_critique
from reflective_qa.data.qasper import QasperSplit

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
CRITIC = CriticConfig(model="claude-sonnet-5-5", effort="medium", max_tokens=100, code_feedback=True)
FINDINGS = [Finding("too_long", "40 words. Use the shortest phrase that answers the question.")]


def _request(critic=CRITIC, findings=FINDINGS):
    (paper,) = QasperSplit.load(FIXTURE).papers
    return build_critic_request(critic, paper, paper.questions[0], list(paper.paragraphs[2:4]), "Foo", ("s1p0",), findings)


def _reply(payload, stop_reason="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], model="claude-sonnet-5-5",
                           stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=2000, output_tokens=90))


def test_the_critic_sees_the_passages_the_question_and_the_draft_with_its_citations():
    prompt = _request()["messages"][0]["content"]
    assert "[s1p0] Experiments. Datasets. We use the Foo dataset." in prompt
    assert "Question: What dataset is used?" in prompt and "Draft answer: Foo\nCited passages: s1p0" in prompt


def test_code_findings_are_shown_only_in_setups_with_code_feedback():
    assert "Automatic checks:\n- too_long: 40 words." in _request()["messages"][0]["content"]
    assert "- none flagged" in _request(findings=[])["messages"][0]["content"]
    quiet = CriticConfig(model="m", effort="low", max_tokens=100, code_feedback=False)
    assert "Automatic checks" not in _request(critic=quiet)["messages"][0]["content"]


def test_the_checklist_is_both_explained_and_enforced_by_the_schema():
    assert all(name in SYSTEM for name in CHECKLIST)
    assert SCHEMA["properties"]["problems"]["items"]["properties"]["check"]["enum"] == list(CHECKLIST)


def test_a_revise_verdict_carries_the_problems_for_the_writer():
    critique = parse_critique("q1", _reply({"verdict": "revise", "problems": [{"check": "too_long", "detail": "Say Foo."}]}))
    assert critique == Critique("q1", "claude-sonnet-5-5", "revise", (Finding("too_long", "Say Foo."),), "ok", 2000, 90)


def test_revise_without_problems_counts_as_pass_and_pass_drops_stray_problems():
    assert parse_critique("q", _reply({"verdict": "revise", "problems": []})).verdict == "pass"
    stray = parse_critique("q", _reply({"verdict": "pass", "problems": [{"check": "too_long", "detail": "x"}]}))
    assert (stray.verdict, stray.problems) == ("pass", ())


def test_unusable_replies_are_errors_that_keep_the_draft():
    assert parse_critique("q", _reply("", "refusal")).verdict == "error"
    assert parse_critique("q", _reply('{"verdict": "rev', "max_tokens")).status == "max_tokens"
    assert parse_critique("q", _reply('{"verdict": "revise"}')).status == "invalid_json"
