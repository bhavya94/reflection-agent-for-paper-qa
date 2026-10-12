import json
from dataclasses import replace
from pathlib import Path

import pytest

from reflective_qa.agent.checks import CodeChecks, Finding
from reflective_qa.agent.critic import CriticConfig
from reflective_qa.agent.reflection import ReflectionAgent, Setup
from reflective_qa.agent.writer import REVISE_INSTRUCTION, RunConfig, Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.experiments.reflection import ReflectionExperiment
from reflective_qa.runs.store import RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
WRITER = Writer(RunConfig(name="w", model="claude-haiku-5-5", k=2, effort="medium", max_tokens=100))
SONNET = CriticConfig("claude-sonnet-5-5", "medium", 100, code_feedback=False)


def _agent(setup, writer=None, max_words=3):
    return ReflectionAgent(setup, writer or Writer.load_all()[setup.writer], CodeChecks(max_words))


def _fingerprint(setup, writer=WRITER, max_words=30):
    return _agent(setup, writer, max_words).fingerprint


def _contexts(tmp_path, setup, answer="Foo"):
    """Drafts for every fixture question, written where the setup's writer run keeps its answers."""
    split = QasperSplit.load(FIXTURE)
    drafts = RunStore(tmp_path).writer_files(Writer.load_all()[setup.writer])[0]
    drafts.write_text("".join(json.dumps({"question_id": q.id, "predicted_answer": answer, "predicted_evidence": ["s1p0"]}) + "\n"
                              for q in split.questions()))
    return ReflectionExperiment.from_config(split, RunStore(tmp_path)).contexts(_agent(setup))


def test_the_five_planned_setups_cover_both_writers():
    setups = Setup.load_all()
    assert {s.writer for s in setups.values()} == {"sonnet-k20", "haiku-k5"}
    assert [n for n, s in setups.items() if s.critic is None] == ["sonnet-code-feedback", "haiku-code-feedback"]
    assert setups["haiku-sonnet-critic-checks"].critic.code_feedback and not setups["haiku-sonnet-critic"].critic.code_feedback


def test_fingerprint_changes_with_the_writer_the_critic_and_any_prompt(monkeypatch):
    setup = Setup("s", "haiku-k5", SONNET)
    base = _fingerprint(setup)
    assert base == _fingerprint(replace(setup, name="renamed"))
    assert base != _fingerprint(setup, Writer(replace(WRITER.config, k=5)))
    assert base != _fingerprint(replace(setup, critic=replace(SONNET, effort="high")))
    monkeypatch.setattr("reflective_qa.agent.critic.SYSTEM", "a different critic prompt")
    assert base != _fingerprint(setup)
    monkeypatch.undo()
    monkeypatch.setattr("reflective_qa.agent.writer.REVISE_INSTRUCTION", "a different revise prompt")
    assert base != _fingerprint(setup)


def test_the_word_limit_counts_only_where_findings_are_shown():
    plain = Setup("s", "haiku-k5", SONNET)
    with_checks = Setup("s", "haiku-k5", replace(SONNET, code_feedback=True))
    code_only = Setup("s", "haiku-k5", None)
    assert _fingerprint(plain, max_words=30) == _fingerprint(plain, max_words=25)
    assert _fingerprint(with_checks, max_words=30) != _fingerprint(with_checks, max_words=25)
    assert _fingerprint(code_only, max_words=30) != _fingerprint(code_only, max_words=25)


def test_a_sampled_question_without_a_draft_is_an_error(tmp_path):
    split = QasperSplit.load(FIXTURE)
    RunStore(tmp_path).writer_files(Writer.load_all()["haiku-k5"])[0].write_text(
        json.dumps({"question_id": "q-extractive", "predicted_answer": "Foo", "predicted_evidence": []}) + "\n")
    with pytest.raises(ValueError, match="no draft for 2 of 3"):
        ReflectionExperiment.from_config(split, RunStore(tmp_path)).contexts(_agent(Setup("s", "haiku-k5", None)))


def test_contexts_carry_the_draft_and_the_check_findings(tmp_path):
    contexts = _contexts(tmp_path, Setup("s", "haiku-k5", None), answer="one two three four")
    assert len(contexts) == 3  # text-only questions
    assert [f.check for f in contexts["q-extractive"].findings] == ["too_long"]


def test_only_setups_with_a_critic_send_critic_requests(tmp_path):
    code_only, with_critic = Setup("s", "haiku-k5", None), Setup("s", "haiku-k5", SONNET)
    assert _agent(code_only).critic_requests(_contexts(tmp_path, code_only)) == {}
    requests = _agent(with_critic).critic_requests(_contexts(tmp_path, with_critic))
    assert len(requests) == 3 and {r["model"] for r in requests.values()} == {"claude-sonnet-5-5"}


def test_the_writer_fixes_critic_problems_or_check_findings_depending_on_the_setup(tmp_path):
    context = _contexts(tmp_path, Setup("s", "haiku-k5", None), answer="one two three four")["q-extractive"]
    assert _agent(Setup("s", "haiku-k5", None)).problems(context, None) == context.findings
    critic = _agent(Setup("s", "haiku-k5", SONNET))
    found = [{"check": "unsupported", "detail": "Not in s1p0."}]
    assert critic.problems(context, {"verdict": "revise", "problems": found}) == [Finding("unsupported", "Not in s1p0.")]
    assert critic.problems(context, {"verdict": "pass", "problems": []}) == []
    assert critic.problems(context, None) == []


def test_the_revision_request_adds_the_draft_and_the_problems_to_the_writers_prompt(tmp_path):
    context = _contexts(tmp_path, Setup("s", "haiku-k5", None))["q-extractive"]
    request = _agent(Setup("s", "haiku-k5", None), WRITER).revise_request(context, [Finding("too_long", "Say Foo.")])
    prompt = request["messages"][0]["content"]
    assert prompt.startswith("Paper: A Tiny Paper") and "Question: What dataset is used?" in prompt
    assert '"answer": "Foo"' in prompt and f"{REVISE_INSTRUCTION}\n- too_long: Say Foo." in prompt
    assert request["model"] == "claude-haiku-5-5" and len(request["messages"]) == 1
