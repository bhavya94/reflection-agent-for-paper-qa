"""Results for the public page, collected from the real run files: M1 baselines and every reflection setup."""

from pathlib import Path

from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.reflection import Context, ReflectionAgent, Setup
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.experiments.baseline import BaselineExperiment
from reflective_qa.experiments.reflection import ReflectionExperiment
from reflective_qa.paths import RUNS_DIR, load_config
from reflective_qa.runs.store import RunStore

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
    store = RunStore(runs_dir)
    baseline, reflection = BaselineExperiment.from_config(split, store), ReflectionExperiment.from_config(split, store)
    checks, writers = CodeChecks.from_config(), Writer.load_all()
    baselines = []
    for name, writer in writers.items():
        r = baseline.report(writer, baseline.requests(writer))
        baselines.append({"name": name, "model": writer.model, "k": writer.k, "answer_f1": r["answer_f1"],
                          "evidence_f1": r["evidence_f1"] if writer.k else None,
                          "unanswerable": r["predicted_unanswerable"], "cost": r["cost"]})
    setups, cases = [], []
    for name, setup in Setup.load_all().items():
        agent = ReflectionAgent(setup, writers[setup.writer], checks)
        contexts = reflection.contexts(agent)
        critiques, revisions = reflection.results(agent)
        if not reflection.has_run(agent, contexts, critiques, revisions):
            raise ValueError(f"{name} has not run with the current code and config; run it before building the page")
        final = ReflectionAgent.final_answers(contexts, critiques, revisions)
        r = reflection.report(contexts, critiques, revisions)
        a = reflection.analyze(contexts, critiques, revisions)
        setups.append({"name": name, "writer": setup.writer, "critic": setup.critic.model if setup.critic else None,
                       "code_feedback": setup.critic is None or setup.critic.code_feedback,
                       "before": r["draft_answer_f1"], "after": r["answer_f1"], "delta": a["delta"], "ci": list(a["ci"]),
                       "evidence_f1": r["evidence_f1"], "unanswerable": r["predicted_unanswerable"],
                       "revised": r["changed"], "better": r["better"], "worse": r["worse"], "cost": r["cost"],
                       "kinds": {k: v | {"label": KIND_LABELS[k]} for k, v in a["kinds"].items()},
                       "contributions": contributions(a, len(contexts))})
        cases.append((name, contexts, final))
    after = {row["name"]: row["after"] for row in setups}
    first, second = load_config("reflection")["analysis"]["replicate_pair"]
    return {"questions": baseline.n, "baselines": baselines, "setups": setups, "examples": pick_examples(cases),
            "spent": store.total_spent(), "noise": {"pair": [first, second], "gap": abs(after[first] - after[second])}}
