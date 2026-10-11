import json
from pathlib import Path
from types import SimpleNamespace

from reflective_qa.answering import RunConfig, load_runs
from reflective_qa.baselines import Costs, read_answers, run_paths, submit_and_collect
from reflective_qa.critic import SYSTEM as CRITIC_SYSTEM, CriticConfig
from reflective_qa.data import load_split
from reflective_qa.reflection import (Setup, build_contexts, critic_requests, critique_record, final_answers, report,
                                      revise_requests, revision_record)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
COSTS = Costs(prices={"claude-sonnet-5-5": {"input": 2.0, "output": 10.0}, "claude-haiku-5-5": {"input": 0.1, "output": 0.5}},
              batch_discount=0.5, assumed_output_tokens=500)
SETUP = Setup("t", "haiku-k5", CriticConfig("claude-sonnet-5-5", "medium", 100, code_feedback=False))


class FakeBatches:
    """Answers critic requests with a verdict per question, and revise requests with a fixed new answer."""

    def __init__(self, verdicts):
        self.verdicts, self.batches, self.retries_off = verdicts, {}, False
        self.messages = SimpleNamespace(batches=SimpleNamespace(create=self._create, results=self._results,
                                                                retrieve=lambda _: SimpleNamespace(processing_status="ended")))

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


def _run(tmp_path, verdicts):
    papers = load_split(FIXTURE)
    writer = load_runs()["haiku-k5"]
    run_paths(writer, tmp_path)[0].write_text("".join(
        json.dumps({"question_id": q.id, "predicted_answer": "a tiny model with many words", "predicted_evidence": ["s0p0"],
                    "output_tokens": 50}) + "\n" for paper in papers for q in paper.questions))
    contexts = build_contexts(SETUP, papers, None, 0, 35, runs_dir=tmp_path)
    client = FakeBatches(verdicts)
    critic_path = submit_and_collect(client, critic_requests(SETUP, contexts), tmp_path / "c.jsonl", tmp_path / "c.batch",
                                     critique_record(SETUP, COSTS), poll_seconds=0)
    critiques = {r["question_id"]: r for r in read_answers(critic_path)}
    revise = revise_requests(SETUP, writer, contexts, critiques)
    revise_path = submit_and_collect(client, revise, tmp_path / "r.jsonl", tmp_path / "r.batch",
                                     revision_record(writer, COSTS), poll_seconds=0)
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


def test_report_compares_final_answers_with_the_drafts(tmp_path):
    papers, contexts, critiques, revisions, _ = _run(tmp_path, {"q-extractive": "revise", "q-abstractive": "revise", "q-boolean": "pass"})
    result = report(papers, contexts, final_answers(contexts, critiques, revisions))
    # q-extractive: "Foo" beats the long draft; q-abstractive: the draft "a tiny model ..." beats "Foo".
    assert (result["revised"], result["better"], result["worse"]) == (2, 1, 1)
    assert result["answer_f1"] != result["draft_answer_f1"]
