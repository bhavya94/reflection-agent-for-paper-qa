"""The public page's data, collected from the real run files: every writer run's one-shot score, and every
reflection setup's change with its interval, its parts and the clearest real example of each kind of revision."""

from reflective_qa.agent.reflection import Context, ReflectionAgent
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.experiments.baseline import BaselineExperiment
from reflective_qa.experiments.reflection import ReflectionExperiment
from reflective_qa.paths import load_config
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
        for r in ReflectionExperiment.before_after(contexts, final):
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


class ResultsCollector:
    def __init__(self, split: QasperSplit, store: RunStore):
        self.store = store
        self.baseline = BaselineExperiment.from_config(split, store)
        self.reflection = ReflectionExperiment.from_config(split, store)

    def collect(self) -> dict:
        """Every number on the page. Refuses a setup with no results for the current code and config, so the page never
        shows an unrun setup as unchanged."""
        baselines = []
        for name, writer in Writer.load_all().items():
            r = self.baseline.report(writer, self.baseline.requests(writer))
            baselines.append({"name": name, "model": writer.model, "k": writer.k, "answer_f1": r["answer_f1"],
                              "evidence_f1": r["evidence_f1"] if writer.k else None,
                              "unanswerable": r["predicted_unanswerable"], "cost": r["cost"]})
        setups, cases = [], []
        for name, agent in ReflectionAgent.load_all().items():
            setup = agent.setup
            contexts = self.reflection.contexts(agent)
            critiques, revisions = self.reflection.results(agent)
            if not self.reflection.has_run(agent, contexts, critiques, revisions):
                raise ValueError(f"{name} has not run with the current code and config; run it before building the page")
            r = self.reflection.report(contexts, critiques, revisions)
            a = self.reflection.analyze(contexts, critiques, revisions)
            setups.append({"name": name, "writer": setup.writer, "critic": setup.critic.model if setup.critic else None,
                           "code_feedback": setup.critic is None or setup.critic.code_feedback,
                           "before": r["draft_answer_f1"], "after": r["answer_f1"], "delta": a["delta"], "ci": list(a["ci"]),
                           "evidence_f1": r["evidence_f1"], "unanswerable": r["predicted_unanswerable"],
                           "revised": r["changed"], "better": r["better"], "worse": r["worse"], "cost": r["cost"],
                           "kinds": {k: v | {"label": KIND_LABELS[k]} for k, v in a["kinds"].items()},
                           "contributions": contributions(a, len(contexts))})
            cases.append((name, contexts, self.reflection.final_answers(contexts, critiques, revisions)))
        after = {row["name"]: row["after"] for row in setups}
        first, second = load_config("reflection")["analysis"]["replicate_pair"]
        return {"questions": self.baseline.n, "baselines": baselines, "setups": setups, "examples": pick_examples(cases),
                "spent": self.store.total_spent(),
                "noise": {"pair": [first, second], "gap": abs(after[first] - after[second])}}
