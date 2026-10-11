"""Results for the public page, collected from the real run files: M1 baselines and every reflection setup."""

import tomllib
from pathlib import Path

from reflective_qa.answering import CONFIG as ANSWERING, load_runs
from reflective_qa.baselines import RUNS_DIR, build_requests, report as baseline_report, total_spent
from reflective_qa.checks import CONFIG as REFLECTION, load_max_words
from reflective_qa.data import Paper
from reflective_qa.reflection import (Context, analyze, build_contexts, final_answers, has_run, load_setups, read_stage,
                                      report, revision_kind, stage_paths)
from reflective_qa.scoring import answer_f1

KIND_LABELS = {"abstain_to_answer": "Unanswerable → answer", "answer_to_abstain": "Answer → Unanswerable",
               "reworded": "Reworded"}


def contributions(analysis: dict, questions: int) -> dict[str, float]:
    """Each kind's share of the total change in mean answer-F1; the shares add up to the setup's change."""
    return {kind: v["n"] * v["mean_delta"] / questions for kind, v in analysis["kinds"].items()}


def pick_examples(cases: list[tuple[str, dict[str, Context], dict[str, dict]]]) -> dict[str, dict]:
    """The clearest real case of each kind across (setup, contexts, final answers): the largest gain, or for
    rewording, the largest loss."""
    best: dict[str, tuple[float, dict]] = {}
    for setup, contexts, final in cases:
        for qid, c in contexts.items():
            before, after = c.draft["predicted_answer"], final[qid]["predicted_answer"]
            if before == after:
                continue
            f1 = (answer_f1(c.question, before)[0], answer_f1(c.question, after)[0])
            delta = f1[1] - f1[0]
            kind = revision_kind(before, after)
            if kind == "reworded" and len(after.split()) < len(before.split()) and delta > 0:
                kind = "shortened"
            strength = -delta if kind == "reworded" else delta
            if strength > 0 and (kind not in best or strength > best[kind][0]):
                best[kind] = (strength, {"question": c.question.text, "before": before, "after": after, "f1": f1, "delta": delta,
                                         "gold": [a.text for a in c.question.answers], "setup": setup})
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
    setups, cases = [], []
    for name, setup in load_setups().items():
        writer = runs[setup.writer]
        contexts = build_contexts(setup, papers, sample["questions"], sample["seed"], max_words, runs_dir)
        paths = stage_paths(setup, writer, max_words, runs_dir)
        critiques, revisions = read_stage(paths, "critic"), read_stage(paths, "revise")
        if not has_run(setup, writer, contexts, critiques, revisions):
            raise ValueError(f"{name} has not run with the current code and config; run it before building the page")
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
        cases.append((name, contexts, final))
    after = {row["name"]: row["after"] for row in setups}
    first, second = analysis_config["replicate_pair"]
    return {"questions": sample["questions"], "baselines": baselines, "setups": setups, "examples": pick_examples(cases),
            "spent": total_spent(runs_dir), "noise": {"pair": [first, second], "gap": abs(after[first] - after[second])}}
