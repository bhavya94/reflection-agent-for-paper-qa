"""Results for the public page, collected from the real run files: M1 baselines and every reflection setup."""

import tomllib
from pathlib import Path

from reflective_qa.answering import CONFIG as ANSWERING, load_runs
from reflective_qa.baselines import RUNS_DIR, build_requests, read_answers, report as baseline_report
from reflective_qa.checks import CONFIG as REFLECTION, load_max_words
from reflective_qa.data import Paper
from reflective_qa.reflection import (Context, analyze, build_contexts, final_answers, load_setups, report,
                                      stage_paths)
from reflective_qa.scoring import answer_f1, is_unanswerable

KIND_LABELS = {"abstain_to_answer": "Unanswerable → answer", "answer_to_abstain": "Answer → Unanswerable",
               "reworded": "Reworded"}


def contributions(analysis: dict, questions: int) -> dict[str, float]:
    """Each kind's share of the total change in mean answer-F1; the shares add up to the setup's change."""
    return {kind: v["n"] * v["mean_delta"] / questions for kind, v in analysis["kinds"].items()}


def pick_examples(contexts: dict[str, Context], final: dict[str, dict]) -> dict[str, dict]:
    """The clearest real case of each kind: the largest gain, or for rewording, the largest loss."""
    best: dict[str, tuple[float, dict]] = {}
    for qid, c in contexts.items():
        before, after = c.draft["predicted_answer"], final[qid]["predicted_answer"]
        if before == after:
            continue
        delta = answer_f1(c.question, after)[0] - answer_f1(c.question, before)[0]
        kind = ("abstain_to_answer" if is_unanswerable(before) and not is_unanswerable(after) else
                "shortened" if len(after.split()) < len(before.split()) and delta > 0 else "reworded")
        strength = -delta if kind == "reworded" else delta
        if kind not in best or strength > best[kind][0]:
            best[kind] = (strength, {"question": c.question.text, "before": before, "after": after, "delta": delta,
                                     "gold": [a.text for a in c.question.answers]})
    return {kind: example for kind, (_, example) in best.items()}


def collect(papers: list[Paper], runs_dir: Path = RUNS_DIR) -> dict:
    sample = tomllib.loads(ANSWERING.read_text())["sample"]
    analysis_config = tomllib.loads(REFLECTION.read_text())["analysis"]
    max_words, runs = load_max_words(), load_runs()
    baselines = []
    for name, run in runs.items():
        r = baseline_report(run, build_requests(run, papers, sample["questions"], sample["seed"]), papers, runs_dir)
        baselines.append({"name": name, "model": run.model, "k": run.k, "answer_f1": r["answer_f1"],
                          "evidence_f1": r["evidence_f1"] if run.k else None,
                          "unanswerable": r["predicted_unanswerable"], "cost": r["cost"]})
    setups, examples = [], {}
    for name, setup in load_setups().items():
        writer = runs[setup.writer]
        contexts = build_contexts(setup, papers, sample["questions"], sample["seed"], max_words, runs_dir)
        paths = stage_paths(setup, writer, max_words, runs_dir)
        critiques = {r["question_id"]: r for r in read_answers(paths["critic"][0])}
        revisions = {r["question_id"]: r for r in read_answers(paths["revise"][0])}
        final = final_answers(contexts, critiques, revisions)
        r = report(papers, contexts, critiques, revisions)
        a = analyze(contexts, final, analysis_config["bootstrap_resamples"], analysis_config["seed"])
        setups.append({"name": name, "writer": setup.writer, "critic": setup.critic.model if setup.critic else None,
                       "code_feedback": setup.critic is None or setup.critic.code_feedback,
                       "before": r["draft_answer_f1"], "after": r["answer_f1"], "delta": a["delta"], "ci": list(a["ci"]),
                       "evidence_f1": r["evidence_f1"], "unanswerable": r["predicted_unanswerable"],
                       "revised": r["changed"], "better": r["better"], "worse": r["worse"], "cost": r["cost"],
                       "kinds": {k: v | {"label": KIND_LABELS[k]} for k, v in a["kinds"].items()},
                       "contributions": contributions(a, len(contexts))})
        for kind, example in pick_examples(contexts, final).items():
            if kind not in examples or abs(example["delta"]) > abs(examples[kind]["delta"]):
                examples[kind] = example | {"setup": name}
    after = {row["name"]: row["after"] for row in setups}
    first, second = analysis_config["replicate_pair"]
    spent = sum(row["cost"] for path in runs_dir.glob("*.jsonl") for row in read_answers(path))
    return {"questions": sample["questions"], "baselines": baselines, "setups": setups, "examples": examples,
            "spent": spent, "noise": {"pair": [first, second], "gap": abs(after[first] - after[second])}}
