"""One round of reflection on a writer run's first answers, for one setup at a time: build each question's context,
estimate and run the critic and revise stages with the Batch API, and score the final answers against the drafts."""

import math

from reflective_qa.agent.checks import Finding
from reflective_qa.agent.critic import Critic
from reflective_qa.agent.reflection import Context, ReflectionAgent
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.analysis import RevisionAnalysis
from reflective_qa.eval.scorer import Prediction, Scorer
from reflective_qa.paths import load_config
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.runs.batch import BatchRunner
from reflective_qa.runs.costs import CostEstimator, Costs
from reflective_qa.runs.store import RunStore


class ReflectionExperiment:
    def __init__(self, split: QasperSplit, store: RunStore, costs: Costs, n: int | None, seed: int,
                 analysis: RevisionAnalysis, runner: BatchRunner | None = None):
        """`runner` is needed only to estimate and run; scoring reads the run files."""
        self.split, self.store, self.costs, self.n, self.seed = split, store, costs, n, seed
        self.analysis, self.runner = analysis, runner
        self.retriever = Retriever(split)

    @classmethod
    def from_config(cls, split: QasperSplit, store: RunStore, n: int | None = None,
                    runner: BatchRunner | None = None) -> "ReflectionExperiment":
        """The writers' seeded sample from configs/answering.toml, or its first n questions."""
        sample, analysis = load_config("answering")["sample"], load_config("reflection")["analysis"]
        return cls(split, store, Costs.from_config(), n or sample["questions"], sample["seed"],
                   RevisionAnalysis(analysis["bootstrap_resamples"], analysis["seed"]), runner)

    def contexts(self, agent: ReflectionAgent) -> dict[str, Context]:
        drafts = {row["question_id"]: row for row in self.store.answers(agent.writer)}
        questions = self.split.sample(self.n, self.seed)
        if missing := [q.id for q in questions if q.id not in drafts]:
            raise ValueError(f"{agent.writer.name} has no draft for {len(missing)} of {len(questions)} sampled questions; "
                             "run its M1 baseline first")
        return {q.id: agent.context(self.split.paper(q.paper_id), q, self.retriever.retrieve(q, agent.writer.k), drafts[q.id])
                for q in questions}

    def results(self, agent: ReflectionAgent) -> tuple[dict[str, dict], dict[str, dict]]:
        """The critic and revise results on disk, keyed by question ID."""
        return self.store.stage(agent, "critic"), self.store.stage(agent, "revise")

    @staticmethod
    def has_run(agent: ReflectionAgent, contexts: dict[str, Context], critiques: dict[str, dict],
                revisions: dict[str, dict]) -> bool:
        """Whether the setup has results for these questions. Code feedback that flags no draft has nothing to run."""
        return bool(set(contexts) & (set(critiques) | set(revisions))) or (
            agent.critic is None and not agent.revise_requests(contexts, {}))

    def estimate(self, agent: ReflectionAgent, contexts: dict[str, Context], critiques: dict[str, dict],
                 revisions: dict[str, dict]) -> dict:
        """Batch cost of what is left to run. Output tokens are measured once a stage has results, else assumed
        (critic) or taken from the drafts (revisions). Before the critic has run, every draft is assumed to need a
        revision."""
        estimator = CostEstimator(self.runner.client, self.costs)
        pending_critic = {q: p for q, p in agent.critic_requests(contexts).items() if q not in critiques}
        stand_in = [Finding("wrong_or_incomplete", "stand-in for an unknown critique")]
        found = {q: agent.problems(c, critiques.get(q)) if (agent.critic is None or q in critiques) else stand_in
                 for q, c in contexts.items()}
        pending_revise = {q: agent.revise_request(contexts[q], f) for q, f in found.items() if f and q not in revisions}
        draft_output = estimator.output_tokens([c.draft for c in contexts.values()])
        critic_dollars = (estimator.dollars(agent.critic.model, pending_critic, estimator.output_tokens(list(critiques.values())))
                          if agent.critic else 0.0)
        revise_dollars = estimator.dollars(agent.writer.model, pending_revise,
                                           estimator.output_tokens(list(revisions.values()), draft_output))
        return {"critic_requests": len(pending_critic), "critic_dollars": critic_dollars,
                "revise_requests": len(pending_revise), "revise_dollars": revise_dollars}

    def run(self, agent: ReflectionAgent, contexts: dict[str, Context], max_dollars: float = math.inf) -> None:
        """Critic stage, then revise stage. Each is refused when its estimate is over max_dollars, unless a batch for
        it is already paid for and waiting to be collected."""
        files = self.store.stage_files(agent)
        if agent.critic:
            e = self.estimate(agent, contexts, *self.results(agent))
            self.runner.run(agent.critic_requests(contexts), files["critic"], self.costs.recorder(Critic.parse, agent.critic.model),
                            estimate=e["critic_dollars"], max_dollars=max_dollars, label=f"{agent.name}: critic")
        critiques, revisions = self.results(agent)
        e = self.estimate(agent, contexts, critiques, revisions)
        self.runner.run(agent.revise_requests(contexts, critiques), files["revise"],
                        self.costs.recorder(Writer.parse, agent.writer.model), estimate=e["revise_dollars"],
                        max_dollars=max_dollars, label=f"{agent.name}: revise")

    def report(self, contexts: dict[str, Context], critiques: dict[str, dict], revisions: dict[str, dict]) -> dict:
        """Final answers scored like M1, plus what reflection did. better and worse compare answer-F1 per question,
        over the answers a revision actually changed."""
        final = ReflectionAgent.final_answers(contexts, critiques, revisions)
        scorer, ids = Scorer(self.split), set(contexts)
        result = scorer.score({qid: Prediction.from_row(row) for qid, row in final.items()}, ids)
        drafts = scorer.score({qid: Prediction.from_row(c.draft) for qid, c in contexts.items()}, ids)
        changed = [r for r in ReflectionAgent.revisions(contexts, final) if r.changed]
        verdicts = [r["verdict"] for r in critiques.values() if r["question_id"] in ids]
        return result | {"draft_answer_f1": drafts["answer_f1"], "critiques": len(verdicts),
                         "critic_revise": verdicts.count("revise"), "critic_errors": verdicts.count("error"),
                         "revisions": sum(qid in ids for qid in revisions), "revision_failures": sum(
                             r["status"] != "ok" for qid, r in revisions.items() if qid in ids),
                         "changed": len(changed), "better": sum(r.delta > 0 for r in changed),
                         "worse": sum(r.delta < 0 for r in changed), "cost": sum(row["cost"] for row in final.values())}

    def analyze(self, contexts: dict[str, Context], critiques: dict[str, dict], revisions: dict[str, dict]) -> dict:
        """The change in answer-F1 with its 95% interval, and the changed answers split by kind."""
        final = ReflectionAgent.final_answers(contexts, critiques, revisions)
        return self.analysis.analyze(ReflectionAgent.revisions(contexts, final))
