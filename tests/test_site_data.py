import json
from pathlib import Path

import pytest

from reflective_qa.baselines import run_paths
from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.reflection import ReflectionAgent, Setup
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.reflection import build_contexts, stage_paths
from reflective_qa.site_data import collect, contributions, pick_examples

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"
DRAFTS = {"q-extractive": "Foo, Bar and a long list of other things", "q-abstractive": "Unanswerable", "q-boolean": "No"}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_drafts(runs_dir: Path) -> None:
    for run in Writer.load_all().values():
        write_rows(run_paths(run, runs_dir)[0], [{"question_id": q, "predicted_answer": a, "predicted_evidence": [],
                                                  "cost": 0.01, "output_tokens": 5} for q, a in DRAFTS.items()])


def test_contributions_add_up_to_the_total_change():
    analysis = {"kinds": {"abstain_to_answer": {"n": 3, "mean_delta": 0.5}, "answer_to_abstain": {"n": 1, "mean_delta": -1.0},
                          "reworded": {"n": 4, "mean_delta": -0.25}}}
    shares = contributions(analysis, questions=10)
    assert shares == pytest.approx({"abstain_to_answer": 0.15, "answer_to_abstain": -0.1, "reworded": -0.1})
    assert sum(shares.values()) == pytest.approx((3 * 0.5 - 1.0 - 4 * 0.25) / 10)  # the mean change over 10 questions


def test_examples_are_the_clearest_case_of_each_kind_across_setups(tmp_path):
    write_drafts(tmp_path)
    agent = ReflectionAgent(Setup("s", "haiku-k5", None), Writer.load_all()["haiku-k5"], CodeChecks(35))
    contexts = build_contexts(agent, QasperSplit.load(FIXTURE), None, 0, runs_dir=tmp_path)
    final = lambda answers: {q: {"predicted_answer": a} for q, a in answers.items()}
    a = final({"q-extractive": "Foo, Bar",                 # shortened: the larger gain
               "q-abstractive": "a model",                 # abstention fixed: the smaller gain
               "q-boolean": "Unanswerable"})               # a new abstention is never shown, even at −1.0
    b = final({"q-extractive": "Foo, Bar and others",      # shortened: the smaller gain
               "q-abstractive": "a tiny model",            # abstention fixed: the larger gain
               "q-boolean": "Not quite, it depends"})      # reworded and worse
    examples = pick_examples([("a", contexts, a), ("b", contexts, b)])
    assert {k: (e["setup"], e["after"]) for k, e in examples.items()} == {
        "shortened": ("a", "Foo, Bar"), "abstain_to_answer": ("b", "a tiny model"),
        "reworded": ("b", "Not quite, it depends")}
    assert examples["reworded"]["delta"] == -1.0


def test_collect_scores_every_setup_and_measures_noise_from_the_replicate_pair(tmp_path):
    write_drafts(tmp_path)
    writers, checks = Writer.load_all(), CodeChecks.from_config()
    for name, setup in Setup.load_all().items():
        paths = stage_paths(ReflectionAgent(setup, writers[setup.writer], checks), tmp_path)
        if setup.critic:
            write_rows(paths["critic"][0], [{"question_id": q, "verdict": "pass", "cost": 0.02} for q in DRAFTS])
        fixed = name == "haiku-sonnet-critic-checks"  # only one of the replicate pair fixes the abstention
        write_rows(paths["revise"][0], [{"question_id": "q-abstractive", "status": "ok", "cost": 0.03,
                                         "predicted_answer": "a tiny model" if fixed else "Unanswerable",
                                         "predicted_evidence": []}])
    data = collect(QasperSplit.load(FIXTURE), tmp_path)
    setups = {row["name"]: row for row in data["setups"]}
    assert setups["haiku-sonnet-critic-checks"]["delta"] == pytest.approx(1 / 3)  # 0 -> 1 on one of three questions
    assert setups["haiku-sonnet-critic"]["delta"] == 0
    assert data["noise"]["gap"] == pytest.approx(1 / 3)
    assert data["examples"]["abstain_to_answer"]["setup"] == "haiku-sonnet-critic-checks"
    critic_setups = sum(bool(s.critic) for s in Setup.load_all().values())
    assert data["spent"] == pytest.approx(3 * 3 * 0.01 + critic_setups * 3 * 0.02 + 6 * 0.03)


def test_collect_refuses_a_setup_that_has_not_run(tmp_path):
    write_drafts(tmp_path)  # M1 drafts only: no critic or revise results, e.g. after a prompt change
    with pytest.raises(ValueError, match="has not run"):
        collect(QasperSplit.load(FIXTURE), tmp_path)
