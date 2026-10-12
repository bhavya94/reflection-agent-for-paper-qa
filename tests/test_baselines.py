import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflective_qa.agent.writer import RunConfig, Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.experiments.baseline import BaselineExperiment
from reflective_qa.runs.batch import BatchRunner, OverBudget
from reflective_qa.runs.costs import Costs
from reflective_qa.runs.store import RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
RUN = Writer(RunConfig(name="test", model="m", k=2, effort="low", max_tokens=100))
COSTS = Costs(prices={"m": {"input": 2.0, "output": 10.0}}, batch_discount=0.5, assumed_output_tokens=500)


def _message(answer: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"answer": answer, "evidence": []}))],
                           model="m", stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


class FakeClient:
    """Batch API stand-in: every request succeeds except IDs in `fail`; `interrupt` makes the first poll raise."""

    def __init__(self, fail=(), interrupt=False):
        self.created, self.fail, self.batches, self.interrupt, self.retries_off = [], set(fail), {}, interrupt, False
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, retrieve=self._retrieve, results=self._results),
            count_tokens=self._count_tokens)

    def with_options(self, **options):
        self.retries_off = options == {"max_retries": 0}
        return self

    def _count_tokens(self, *, model, system, messages, output_config):  # the real endpoint rejects max_tokens
        return SimpleNamespace(input_tokens=1000)

    def _create(self, requests):
        assert self.retries_off, "a retried create could submit a second paid batch"
        self.retries_off = False
        batch_id = f"batch_{len(self.created) + 1}"
        self.created.append([r["custom_id"] for r in requests])
        self.batches[batch_id] = [r["custom_id"] for r in requests]
        return SimpleNamespace(id=batch_id)

    def _retrieve(self, batch_id):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        return SimpleNamespace(processing_status="ended")

    def _results(self, batch_id):
        for qid in self.batches[batch_id]:
            kind = "errored" if qid in self.fail else "succeeded"
            yield SimpleNamespace(custom_id=qid, result=SimpleNamespace(type=kind, message=_message("Foo")))


def _experiment(tmp_path, client=None, n=None):
    return BaselineExperiment(QasperSplit.load(FIXTURE), RunStore(tmp_path), COSTS, n, seed=0,
                              runner=BatchRunner(client or FakeClient(), poll_seconds=0))


def _requests(run=RUN, n=None):
    return _experiment(Path("unused"), n=n).requests(run)


def _ids(path):
    return [r["question_id"] for r in RunStore.read(path)]


def test_samples_are_seeded_and_smaller_samples_are_prefixes_of_larger_ones():
    assert list(_requests(n=3))[:2] == list(_requests(n=2))
    assert sorted(_requests()) == ["q-abstractive", "q-boolean", "q-extractive"]


def test_only_unanswered_questions_are_submitted_and_failures_stay_pending(tmp_path):
    client = FakeClient(fail={"q-boolean"})
    path = _experiment(tmp_path, client).run(RUN, _requests())
    assert sorted(_ids(path)) == ["q-abstractive", "q-extractive"]
    _experiment(tmp_path, client).run(RUN, _requests())
    assert client.created[1] == ["q-boolean"]


def test_an_interrupted_run_resumes_its_saved_batch_without_paying_again(tmp_path):
    client = FakeClient(interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        _experiment(tmp_path, client).run(RUN, _requests())
    assert RunStore(tmp_path).writer_files(RUN)[1].exists()  # batch ID saved before waiting
    path = _experiment(tmp_path, client).run(RUN, _requests())
    assert len(client.created) == 1 and sorted(_ids(path)) == sorted(_requests())


def test_resuming_never_writes_a_question_twice(tmp_path):
    client = FakeClient()
    client.batches["batch_saved"] = list(_requests())  # the saved batch also holds an answered question
    answers, batch = RunStore(tmp_path).writer_files(RUN)
    answers.write_text(json.dumps({"question_id": "q-extractive", "cost": 0}) + "\n")
    batch.write_text("batch_saved")
    _experiment(tmp_path, client).run(RUN, _requests())
    assert sorted(_ids(answers)) == sorted(_requests())


def test_a_changed_config_starts_a_new_file_instead_of_mixing_answers(tmp_path):
    client = FakeClient()
    first = _experiment(tmp_path, client).run(RUN, _requests())
    changed = Writer(replace(RUN.config, effort="high"))
    second = _experiment(tmp_path, client).run(changed, _requests(changed))
    assert first != second and len(client.created) == 2 and len(_ids(first)) == len(_ids(second)) == 3


def test_each_answer_records_its_cost_and_estimates_switch_to_measured_output(tmp_path):
    before = _experiment(tmp_path).estimate(RUN, _requests())
    assert before["output_tokens"] == 500 and before["dollars"] == (3 * 1000 * 2.0 + 3 * 500 * 10.0) / 1e6 * 0.5
    path = _experiment(tmp_path).run(RUN, _requests())
    assert RunStore.read(path)[0]["cost"] == (1000 * 2.0 + 100 * 10.0) / 1e6 * 0.5
    after = _experiment(tmp_path).estimate(RUN, _requests())
    assert after["output_tokens"] == 100 and after["output_measured"]


def test_report_scores_a_run_on_its_own_sample_with_its_cost(tmp_path):
    _experiment(tmp_path).run(RUN, _requests(n=2))
    result = _experiment(tmp_path).report(RUN, _requests(n=2))
    assert (result["questions"], result["missing"]) == (2, 0)  # the third fixture question is outside the sample
    assert result["cost"] == 2 * (1000 * 2.0 + 100 * 10.0) / 1e6 * 0.5 and result["output_tokens"] == 100


def test_a_run_over_budget_is_refused_unless_its_batch_is_already_paid_for(tmp_path):
    client = FakeClient()
    with pytest.raises(OverBudget, match=r"test: estimate \$2.00 is over --max-dollars 1.0"):
        _experiment(tmp_path, client).run(RUN, _requests(), estimate=2.0, max_dollars=1.0)
    assert client.created == []
    client.batches["batch_paid"] = list(_requests())
    answers, batch = RunStore(tmp_path).writer_files(RUN)
    batch.write_text("batch_paid")
    _experiment(tmp_path, client).run(RUN, _requests(), estimate=2.0, max_dollars=1.0)
    assert client.created == [] and sorted(_ids(answers)) == sorted(_requests())
