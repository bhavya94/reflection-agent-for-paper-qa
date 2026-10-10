import json
from pathlib import Path
from types import SimpleNamespace

from reflective_qa.answering import RunConfig
from reflective_qa.baselines import Costs, build_requests, estimate, read_answers, run_batch
from reflective_qa.data import load_split

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
RUN = RunConfig(name="test", model="m", k=2, effort="low", max_tokens=100)
COSTS = Costs(prices={"m": {"input": 2.0, "output": 10.0}}, batch_discount=0.5, assumed_output_tokens=500)


def _message(answer: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"answer": answer, "evidence": []}))],
                           stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


class FakeClient:
    """Batch API stand-in: every submitted request succeeds, except IDs listed in `fail`."""

    def __init__(self, fail=()):
        self.created, self.fail, self.requests = [], set(fail), {}
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, retrieve=lambda _: SimpleNamespace(processing_status="ended"),
                                    results=self._results),
            count_tokens=lambda **params: SimpleNamespace(input_tokens=1000))

    def _create(self, requests):
        self.created.append([r["custom_id"] for r in requests])
        self.requests[f"batch_{len(self.created)}"] = [r["custom_id"] for r in requests]
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def _results(self, batch_id):
        for qid in self.requests[batch_id]:
            kind = "errored" if qid in self.fail else "succeeded"
            yield SimpleNamespace(custom_id=qid, result=SimpleNamespace(type=kind, message=_message("Foo")))


def _requests():
    return build_requests(RUN, load_split(FIXTURE))


def test_requests_cover_the_text_only_questions():
    assert list(_requests()) == ["q-extractive", "q-abstractive", "q-boolean"]


def test_only_unanswered_questions_are_submitted_and_failures_stay_pending(tmp_path):
    client = FakeClient(fail={"q-boolean"})
    path = run_batch(client, RUN, _requests(), tmp_path, poll_seconds=0)
    assert [r["question_id"] for r in read_answers(path)] == ["q-extractive", "q-abstractive"]
    run_batch(client, RUN, _requests(), tmp_path, poll_seconds=0)
    assert client.created[1] == ["q-boolean"]


def test_an_interrupted_run_resumes_its_batch_instead_of_paying_again(tmp_path):
    client = FakeClient()
    client.requests["batch_saved"] = list(_requests())
    (tmp_path / "test.batch").write_text("batch_saved")
    path = run_batch(client, RUN, _requests(), tmp_path, poll_seconds=0)
    assert client.created == [] and len(read_answers(path)) == 3
    assert not (tmp_path / "test.batch").exists()


def test_a_finished_run_makes_no_calls(tmp_path):
    client = FakeClient()
    run_batch(client, RUN, _requests(), tmp_path, poll_seconds=0)
    run_batch(client, RUN, _requests(), tmp_path, poll_seconds=0)
    assert len(client.created) == 1


def test_estimate_uses_measured_output_tokens_once_a_run_has_answers(tmp_path):
    before = estimate(FakeClient(), RUN, _requests(), COSTS, out_dir=tmp_path)
    assert before["output_tokens"] == 500 and before["dollars"] == (3 * 1000 * 2.0 + 3 * 500 * 10.0) / 1e6 * 0.5
    run_batch(FakeClient(), RUN, _requests(), tmp_path, poll_seconds=0)
    after = estimate(FakeClient(), RUN, _requests(), COSTS, out_dir=tmp_path)
    assert after["output_tokens"] == 100 and after["output_measured"]
