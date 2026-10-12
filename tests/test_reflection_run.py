import json
from pathlib import Path
from types import SimpleNamespace

from reflective_qa.answering import RunConfig, load_runs
import pytest

from reflective_qa.baselines import Costs, answer_record, read_answers, run_paths, submit_and_collect
from reflective_qa.critic import SYSTEM as CRITIC_SYSTEM, CriticConfig
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.answering import build_request, parse_message
from reflective_qa.reflection import (Setup, build_contexts, critic_requests, critique_record, estimate_setup,
                                      final_answers, report, revise_request, revise_requests, run_setup)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
COSTS = Costs(prices={"claude-sonnet-5-5": {"input": 2.0, "output": 10.0}, "claude-haiku-5-5": {"input": 0.1, "output": 0.5}},
              batch_discount=0.5, assumed_output_tokens=500)
SETUP = Setup("t", "haiku-k5", CriticConfig("claude-sonnet-5-5", "medium", 100, code_feedback=False))


class FakeBatches:
    """Answers critic requests with a verdict per question, and revise requests with a fixed new answer."""

    def __init__(self, verdicts, input_tokens=1000):
        self.verdicts, self.batches, self.retries_off = verdicts, {}, False
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, results=self._results,
                                    retrieve=lambda _: SimpleNamespace(processing_status="ended")),
            count_tokens=lambda *, model, system, messages, output_config: SimpleNamespace(input_tokens=input_tokens))

    def with_options(self, **options):
        self.retries_off = options == {"max_retries": 0}
        return self

    def _create(self, requests):
        assert self.retries_off
        batch_id = f"b{len(self.batches)}"
        self.batches[batch_id] = [(r["custom_id"], r["params"]["system"] == CRITIC_SYSTEM) for r in requests]
        return SimpleNamespace(id=batch_id)

    def _results(self, batch_id):
        for qid, is_critic in self.batches[batch_id]:
            payload = ({"verdict": self.verdicts[qid], "problems": [{"check": "too_long", "detail": "Shorten."}]}
                       if is_critic else {"answer": "Foo", "evidence": ["s1p0"]})
            message = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))], model="m",
                                      stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=100))
            yield SimpleNamespace(custom_id=qid, result=SimpleNamespace(type="succeeded", message=message))


def _setup_drafts(tmp_path):
    split = QasperSplit.load(FIXTURE)
    writer = load_runs()["haiku-k5"]
    run_paths(writer, tmp_path)[0].write_text("".join(
        json.dumps({"question_id": q.id, "predicted_answer": "a tiny model with many words", "predicted_evidence": ["s0p0"],
                    "output_tokens": 50}) + "\n" for q in split.questions()))
    return split, writer, build_contexts(SETUP, split, None, 0, 35, runs_dir=tmp_path)


def _paths(tmp_path):
    return {stage: (tmp_path / f"{stage}.jsonl", tmp_path / f"{stage}.batch") for stage in ("critic", "revise")}


def _run(tmp_path, verdicts):
    papers, writer, contexts = _setup_drafts(tmp_path)
    client = FakeBatches(verdicts)
    critic_path = submit_and_collect(client, critic_requests(SETUP, contexts), tmp_path / "c.jsonl", tmp_path / "c.batch",
                                     critique_record(SETUP, COSTS), poll_seconds=0)
    critiques = {r["question_id"]: r for r in read_answers(critic_path)}
    revise = revise_requests(SETUP, writer, contexts, critiques)
    revise_path = submit_and_collect(client, revise, tmp_path / "r.jsonl", tmp_path / "r.batch",
                                     lambda qid, m: answer_record(parse_message(qid, m), writer, COSTS), poll_seconds=0)
    revisions = {r["question_id"]: r for r in read_answers(revise_path)}
    return papers, contexts, critiques, revisions, revise


def test_only_drafts_the_critic_flags_are_revised(tmp_path):
    _, _, critiques, revisions, revise = _run(tmp_path, {"q-extractive": "revise", "q-abstractive": "pass", "q-boolean": "revise"})
    assert sorted(revise) == sorted(revisions) == ["q-boolean", "q-extractive"]
    assert critiques["q-abstractive"]["verdict"] == "pass"


def test_final_answers_use_the_revision_and_count_both_stages_cost(tmp_path):
    _, contexts, critiques, revisions, _ = _run(tmp_path, {"q-extractive": "revise", "q-abstractive": "pass", "q-boolean": "pass"})
    final = final_answers(contexts, critiques, revisions)
    assert (final["q-extractive"]["predicted_answer"], final["q-extractive"]["revised"]) == ("Foo", True)
    assert final["q-abstractive"]["predicted_answer"] == "a tiny model with many words"
    critic_cost, revise_cost = (1000 * 2.0 + 100 * 10.0) / 1e6 * 0.5, (1000 * 0.1 + 100 * 0.5) / 1e6 * 0.5
    assert final["q-extractive"]["cost"] == critic_cost + revise_cost and final["q-abstractive"]["cost"] == critic_cost


def test_a_failed_revision_keeps_the_draft(tmp_path):
    _, contexts, critiques, revisions, _ = _run(tmp_path, {"q-extractive": "revise", "q-abstractive": "pass", "q-boolean": "pass"})
    revisions["q-extractive"]["status"] = "refusal"
    assert final_answers(contexts, critiques, revisions)["q-extractive"]["revised"] is False


def test_report_counts_changed_answers_and_which_way_answer_f1_moved(tmp_path):
    # q-extractive: the long draft scores 0 against "Foo, Bar"; the revision "Foo" scores higher -> better.
    # q-boolean: draft and revision both score 0 against "No" -> changed, but neither better nor worse.
    papers, contexts, critiques, revisions, _ = _run(tmp_path, {"q-extractive": "revise", "q-abstractive": "pass", "q-boolean": "revise"})
    r = report(papers, contexts, critiques, revisions)
    assert (r["critiques"], r["critic_revise"], r["revisions"], r["changed"], r["better"], r["worse"]) == (3, 2, 2, 2, 1, 0)
    assert r["answer_f1"] > r["draft_answer_f1"]


def test_critic_over_budget_is_refused_before_any_batch(tmp_path):
    papers, writer, contexts = _setup_drafts(tmp_path)
    client = FakeBatches({}, input_tokens=10_000_000)
    with pytest.raises(SystemExit, match="critic estimate"):
        run_setup(client, SETUP, writer, contexts, COSTS, _paths(tmp_path), max_dollars=1.0, poll_seconds=0)
    assert client.batches == {}


def test_revise_over_budget_is_refused_after_the_critic_stage(tmp_path):
    papers, writer, contexts = _setup_drafts(tmp_path)
    client = FakeBatches({q: "revise" for q in contexts})
    # Sonnet reviews count 1,000 tokens (about $0.01 for the stage); Haiku revisions count 10M (about $1.50).
    client.messages.count_tokens = lambda *, model, **_: SimpleNamespace(
        input_tokens=10_000_000 if model == "claude-haiku-5-5" else 1000)
    with pytest.raises(SystemExit, match="revise estimate"):
        run_setup(client, SETUP, writer, contexts, COSTS, _paths(tmp_path), max_dollars=0.05, poll_seconds=0)
    assert len(client.batches) == 1 and len(read_answers(_paths(tmp_path)["critic"][0])) == 3


def test_a_paid_batch_waiting_to_be_collected_is_never_blocked(tmp_path):
    papers, writer, contexts = _setup_drafts(tmp_path)
    client = FakeBatches({q: "pass" for q in contexts}, input_tokens=10_000_000)
    client.batches["b-paid"] = [(q, True) for q in contexts]
    paths = _paths(tmp_path)
    paths["critic"][1].write_text("b-paid")
    run_setup(client, SETUP, writer, contexts, COSTS, paths, max_dollars=0.01, poll_seconds=0)
    assert len(read_answers(paths["critic"][0])) == 3 and list(client.batches) == ["b-paid"]


def test_the_estimate_stops_assuming_revisions_once_the_critic_passes_drafts(tmp_path):
    papers, writer, contexts = _setup_drafts(tmp_path)
    client = FakeBatches({})
    before = estimate_setup(client, SETUP, writer, contexts, COSTS, {}, {})
    passed = {q: {"question_id": q, "verdict": "pass", "problems": [], "output_tokens": 80} for q in contexts}
    after = estimate_setup(client, SETUP, writer, contexts, COSTS, passed, {})
    assert (before["critic_requests"], before["revise_requests"]) == (3, 3)
    assert (after["critic_requests"], after["revise_requests"], after["revise_dollars"]) == (0, 0, 0.0)


def test_revisions_keep_the_writer_prompt_and_schema_and_the_critic_sees_every_passage(tmp_path):
    papers, writer, contexts = _setup_drafts(tmp_path)
    c = contexts["q-extractive"]
    original = build_request(writer, c.paper, c.question, c.passages)
    revised = revise_request(writer, c, [])
    assert {k: v for k, v in revised.items() if k != "messages"} == {k: v for k, v in original.items() if k != "messages"}
    prompt = critic_requests(SETUP, {"q": c})["q"]["messages"][0]["content"]
    assert all(f"[{p.id}]" in prompt for p in c.passages)
