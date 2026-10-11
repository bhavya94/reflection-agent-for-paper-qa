import json
from pathlib import Path

import pytest

from reflective_qa.answering import load_runs
from reflective_qa.baselines import run_paths
from reflective_qa.data import load_split
from reflective_qa.reflection import Setup, build_contexts
from reflective_qa.site_data import contributions, pick_examples

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


def test_contributions_add_up_to_the_total_change():
    analysis = {"kinds": {"abstain_to_answer": {"n": 3, "mean_delta": 0.5}, "answer_to_abstain": {"n": 1, "mean_delta": -1.0},
                          "reworded": {"n": 4, "mean_delta": -0.25}}}
    shares = contributions(analysis, questions=10)
    assert shares == pytest.approx({"abstain_to_answer": 0.15, "answer_to_abstain": -0.1, "reworded": -0.1})
    assert sum(shares.values()) == pytest.approx((3 * 0.5 - 1.0 - 4 * 0.25) / 10)  # the mean change over 10 questions


def test_examples_show_the_clearest_case_of_each_kind(tmp_path):
    papers = load_split(FIXTURE)
    drafts = {"q-extractive": "Foo, Bar and a long list of other things", "q-abstractive": "Unanswerable", "q-boolean": "No"}
    run_paths(load_runs()["haiku-k5"], tmp_path)[0].write_text("".join(
        json.dumps({"question_id": q, "predicted_answer": a, "predicted_evidence": []}) + "\n" for q, a in drafts.items()))
    contexts = build_contexts(Setup("s", "haiku-k5", None), papers, None, 0, 35, runs_dir=tmp_path)
    final = {"q-extractive": {"predicted_answer": "Foo, Bar"},              # shorter and better
             "q-abstractive": {"predicted_answer": "a tiny model"},         # abstention fixed
             "q-boolean": {"predicted_answer": "Not quite, it depends"}}    # reworded and worse
    examples = pick_examples(contexts, final)
    assert set(examples) == {"shortened", "abstain_to_answer", "reworded"}
    assert examples["abstain_to_answer"]["after"] == "a tiny model" and examples["reworded"]["delta"] == -1.0
