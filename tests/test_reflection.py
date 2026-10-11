import json
from dataclasses import replace
from pathlib import Path

from reflective_qa.answering import RunConfig, load_runs
from reflective_qa.baselines import run_paths
from reflective_qa.checks import Finding
from reflective_qa.critic import CriticConfig
from reflective_qa.data import load_split
from reflective_qa.reflection import (REVISE_INSTRUCTION, Setup, build_contexts, critic_requests, fingerprint,
                                      load_setups, problems, revise_request)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
WRITER = RunConfig(name="w", model="claude-haiku-5-5", k=2, effort="medium", max_tokens=100)
SONNET = CriticConfig("claude-sonnet-5-5", "medium", 100, code_feedback=False)


def _contexts(tmp_path, setup, answer="Foo"):
    """Drafts for every fixture question, written where the setup's writer run keeps its answers."""
    papers = load_split(FIXTURE)
    drafts = run_paths(load_runs()[setup.writer], tmp_path)[0]
    drafts.write_text("".join(json.dumps({"question_id": q.id, "predicted_answer": answer, "predicted_evidence": ["s1p0"]}) + "\n"
                              for paper in papers for q in paper.questions))
    return build_contexts(setup, papers, n=None, seed=0, max_words=3, runs_dir=tmp_path)


def test_the_five_planned_setups_cover_both_writers():
    setups = load_setups()
    assert {s.writer for s in setups.values()} == {"sonnet-k20", "haiku-k5"}
    assert [n for n, s in setups.items() if s.critic is None] == ["sonnet-code-feedback", "haiku-code-feedback"]
    assert setups["haiku-sonnet-critic-checks"].critic.code_feedback and not setups["haiku-sonnet-critic"].critic.code_feedback


def test_fingerprint_changes_with_the_critic_the_writer_and_the_word_limit():
    setup = Setup("s", "haiku-k5", SONNET)
    base = fingerprint(setup, WRITER, 30)
    assert base == fingerprint(replace(setup, name="renamed"), WRITER, 30)
    assert base != fingerprint(replace(setup, critic=replace(SONNET, code_feedback=True)), WRITER, 30)
    assert base != fingerprint(setup, replace(WRITER, k=5), 30) != fingerprint(setup, WRITER, 25)


def test_contexts_carry_the_draft_and_the_check_findings(tmp_path):
    contexts = _contexts(tmp_path, Setup("s", "haiku-k5", None), answer="one two three four")
    assert len(contexts) == 3  # text-only questions
    assert [f.check for f in contexts["q-extractive"].findings] == ["too_long"]


def test_only_setups_with_a_critic_send_critic_requests(tmp_path):
    assert critic_requests(Setup("s", "haiku-k5", None), _contexts(tmp_path, Setup("s", "haiku-k5", None))) == {}
    requests = critic_requests(Setup("s", "haiku-k5", SONNET), _contexts(tmp_path, Setup("s", "haiku-k5", SONNET)))
    assert len(requests) == 3 and {r["model"] for r in requests.values()} == {"claude-sonnet-5-5"}


def test_the_writer_fixes_critic_problems_or_check_findings_depending_on_the_setup(tmp_path):
    context = _contexts(tmp_path, Setup("s", "haiku-k5", None), answer="one two three four")["q-extractive"]
    assert problems(Setup("s", "haiku-k5", None), context, None) == context.findings
    critic_setup = Setup("s", "haiku-k5", SONNET)
    found = [{"check": "unsupported", "detail": "Not in s1p0."}]
    assert problems(critic_setup, context, {"verdict": "revise", "problems": found}) == [Finding("unsupported", "Not in s1p0.")]
    assert problems(critic_setup, context, {"verdict": "pass", "problems": []}) == []
    assert problems(critic_setup, context, None) == []


def test_the_revision_request_adds_the_draft_and_the_problems_to_the_writers_prompt(tmp_path):
    context = _contexts(tmp_path, Setup("s", "haiku-k5", None))["q-extractive"]
    request = revise_request(WRITER, context, [Finding("too_long", "Say Foo.")])
    prompt = request["messages"][0]["content"]
    assert prompt.startswith("Paper: A Tiny Paper") and "Question: What dataset is used?" in prompt
    assert '"answer": "Foo"' in prompt and f"{REVISE_INSTRUCTION}\n- too_long: Say Foo." in prompt
    assert request["model"] == "claude-haiku-5-5" and len(request["messages"]) == 1
