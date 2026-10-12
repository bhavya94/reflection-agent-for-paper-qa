"""Results for the public page, collected from the real run files: M1 baselines and every reflection setup."""

from pathlib import Path

from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.reflection import Context, ReflectionAgent, Setup
from reflective_qa.agent.writer import Writer
from reflective_qa.baselines import RUNS_DIR, build_requests, report as baseline_report, total_spent
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.analysis import RevisionAnalysis
from reflective_qa.paths import load_config
from reflective_qa.reflection import build_contexts, has_run, read_stage, report, stage_paths

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
        for r in ReflectionAgent.revisions(contexts, final):
            if not r.changed:
                continue
            kind = r.kind
            if kind == "reworded" and len(r.after.split()) < len(r.before.split()) and r.delta > 0:
                kind = "shortened"
            strength = -r.delta if kind == "reworded" else r.delta
            if strength > 0 and (kind not in best or strength > best[kind][0]):
                best[kind] = (strength, {"question": r.question.text, "before": r.before, "after": r.after, "f1": r.f1,
                                         "delta": r.delta, "gold": [a.text for a in r.question.answers], "setup": setup})
    return {kind: example for kind, (_, example) in best.items()}


def collect(split: QasperSplit, runs_dir: Path = RUNS_DIR) -> dict:
    sample, analysis_config = load_config("answering")["sample"], load_config("reflection")["analysis"]
    checks, writers = CodeChecks.from_config(), Writer.load_all()
    baselines = []
    for name, writer in writers.items():
        r = baseline_report(writer, build_requests(writer, split, sample["questions"], sample["seed"]), split, runs_dir)
        baselines.append({"name": name, "model": writer.model, "k": writer.k, "answer_f1": r["answer_f1"],
                          "evidence_f1": r["evidence_f1"] if writer.k else None,
                          "unanswerable": r["predicted_unanswerable"], "cost": r["cost"]})
    setups, cases = [], []
    for name, setup in Setup.load_all().items():
        agent = ReflectionAgent(setup, writers[setup.writer], checks)
        contexts = build_contexts(agent, split, sample["questions"], sample["seed"], runs_dir)
        paths = stage_paths(agent, runs_dir)
        critiques, revisions = read_stage(paths, "critic"), read_stage(paths, "revise")
        if not has_run(agent, contexts, critiques, revisions):
            raise ValueError(f"{name} has not run with the current code and config; run it before building the page")
        final = ReflectionAgent.final_answers(contexts, critiques, revisions)
        r = report(split, contexts, critiques, revisions)
        a = RevisionAnalysis(analysis_config["bootstrap_resamples"], analysis_config["seed"]).analyze(
            ReflectionAgent.revisions(contexts, final))
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
