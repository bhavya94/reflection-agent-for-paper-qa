"""One round of reflection on the M1 drafts: code checks and an optional critic flag problems, and the writer revises
the flagged drafts once. Each setup's files are named with a fingerprint of everything that shapes its requests."""

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from reflective_qa.agent.checks import CodeChecks, Finding
from reflective_qa.agent.critic import Critic
from reflective_qa.agent.reflection import Context, ReflectionAgent, Setup
from reflective_qa.agent.writer import Writer
from reflective_qa.baselines import (RUNS_DIR, Costs, answer_record, count_input_tokens, load_costs, read_answers, run_paths,
                                     submit_and_collect)
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.analysis import RevisionAnalysis
from reflective_qa.eval.scorer import Prediction, Scorer
from reflective_qa.paths import DEV_SPLIT, load_config
from reflective_qa.retrieval.bm25 import Retriever


def stage_paths(agent: ReflectionAgent, out_dir: Path = RUNS_DIR) -> dict[str, tuple[Path, Path]]:
    """(answers, batch) file pairs for the critic and revise stages."""
    stem = f"reflect-{agent.name}-{agent.fingerprint}"
    return {stage: (out_dir / f"{stem}.{stage}.jsonl", out_dir / f"{stem}.{stage}.batch") for stage in ("critic", "revise")}


def read_stage(paths: dict[str, tuple[Path, Path]], stage: str) -> dict[str, dict]:
    """One stage's results on disk, keyed by question ID."""
    return {r["question_id"]: r for r in read_answers(paths[stage][0])}


def build_contexts(agent: ReflectionAgent, split: QasperSplit, n: int | None, seed: int,
                   runs_dir: Path = RUNS_DIR) -> dict[str, Context]:
    drafts = {row["question_id"]: row for row in read_answers(run_paths(agent.writer, runs_dir)[0])}
    questions = split.sample(n, seed)
    if missing := [q.id for q in questions if q.id not in drafts]:
        raise ValueError(f"{agent.writer.name} has no draft for {len(missing)} of {len(questions)} sampled questions; "
                         "run its M1 baseline first")
    retriever = Retriever(split)
    contexts = {}
    for question in questions:
        paper = split.paper(question.paper_id)
        contexts[question.id] = agent.context(paper, question, retriever.retrieve(question, agent.writer.k), drafts[question.id])
    return contexts








def critique_record(agent: ReflectionAgent, costs: Costs) -> Callable[[str, Any], dict]:
    def record(qid: str, message: Any) -> dict:
        c = Critic.parse(qid, message)
        return {"question_id": qid, "verdict": c.verdict, "problems": [asdict(f) for f in c.problems], "status": c.status,
                "model": c.model, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "cost": costs.dollars(agent.critic.model, c.input_tokens, c.output_tokens)}
    return record


def has_run(agent: ReflectionAgent, contexts: dict[str, Context], critiques: dict[str, dict],
            revisions: dict[str, dict]) -> bool:
    """Whether the setup has results for these questions. Code feedback that flags no draft has nothing to run."""
    return bool(set(contexts) & (set(critiques) | set(revisions))) or (
        agent.critic is None and not agent.revise_requests(contexts, {}))


def report(split: QasperSplit, contexts: dict[str, Context], critiques: dict[str, dict],
           revisions: dict[str, dict]) -> dict:
    """Final answers scored like M1, plus what reflection did. better and worse compare answer-F1 per question,
    over the answers a revision actually changed."""
    final = ReflectionAgent.final_answers(contexts, critiques, revisions)
    scorer, ids = Scorer(split), set(contexts)
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


def _mean_output(rows: list[dict], default: float) -> float:
    return statistics.mean(r["output_tokens"] for r in rows) if rows else default


def _dollars(client: Any, model: str, requests: dict[str, dict], costs: Costs, output_tokens: float) -> float:
    if not requests:
        return 0.0
    return costs.dollars(model, count_input_tokens(client, requests) * len(requests), output_tokens * len(requests))


def estimate_setup(client: Any, agent: ReflectionAgent, contexts: dict[str, Context], costs: Costs,
                   critiques: dict[str, dict], revisions: dict[str, dict]) -> dict:
    """Batch cost of what is left to run. Output tokens are measured once a stage has results, else assumed (critic)
    or taken from the drafts (revisions). Before the critic has run, every draft is assumed to need a revision."""
    pending_critic = {q: p for q, p in agent.critic_requests(contexts).items() if q not in critiques}
    stand_in = [Finding("wrong_or_incomplete", "stand-in for an unknown critique")]
    found = {q: agent.problems(c, critiques.get(q)) if (agent.critic is None or q in critiques) else stand_in
             for q, c in contexts.items()}
    pending_revise = {q: agent.revise_request(contexts[q], f) for q, f in found.items() if f and q not in revisions}
    draft_output = _mean_output([c.draft for c in contexts.values()], costs.assumed_output_tokens)
    critic_dollars = (_dollars(client, agent.critic.model, pending_critic, costs,
                               _mean_output(list(critiques.values()), costs.assumed_output_tokens)) if agent.critic else 0.0)
    revise_dollars = _dollars(client, agent.writer.model, pending_revise, costs,
                              _mean_output(list(revisions.values()), draft_output))
    return {"critic_requests": len(pending_critic), "critic_dollars": critic_dollars,
            "revise_requests": len(pending_revise), "revise_dollars": revise_dollars}


def run_setup(client: Any, agent: ReflectionAgent, contexts: dict[str, Context], costs: Costs,
              paths: dict[str, tuple[Path, Path]], max_dollars: float, poll_seconds: float = 30) -> None:
    """Critic stage, then revise stage. Each is refused when its estimate is over max_dollars, unless a batch for it
    is already paid for and waiting to be collected."""
    read = lambda stage: read_stage(paths, stage)
    if agent.critic:
        e = estimate_setup(client, agent, contexts, costs, read("critic"), read("revise"))
        if e["critic_dollars"] > max_dollars and not paths["critic"][1].exists():
            raise SystemExit(f"{agent.name}: critic estimate ${e['critic_dollars']:.2f} is over --max-dollars {max_dollars}")
        submit_and_collect(client, agent.critic_requests(contexts), *paths["critic"], critique_record(agent, costs),
                           poll_seconds)
    critiques = read("critic")
    e = estimate_setup(client, agent, contexts, costs, critiques, read("revise"))
    if e["revise_dollars"] > max_dollars and not paths["revise"][1].exists():
        raise SystemExit(f"{agent.name}: revise estimate ${e['revise_dollars']:.2f} is over --max-dollars {max_dollars}")
    record = lambda qid, message: answer_record(Writer.parse(qid, message), agent.writer, costs)
    submit_and_collect(client, agent.revise_requests(contexts, critiques), *paths["revise"], record, poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "run", "report"])
    parser.add_argument("setups", nargs="*", help="setup names from configs/reflection.toml; default: all")
    parser.add_argument("--dev", type=Path, default=DEV_SPLIT)
    parser.add_argument("--questions", type=int, help="size of the M1 question sample; default: the config's")
    parser.add_argument("--max-dollars", type=float, help="required for run: per stage, refuse if the estimate is higher")
    args = parser.parse_args()
    if args.command == "run" and args.max_dollars is None:
        raise SystemExit("run needs --max-dollars")
    sample = load_config("answering")["sample"]
    split, costs, writers, checks = QasperSplit.load(args.dev), load_costs(), Writer.load_all(), CodeChecks.from_config()
    analysis = load_config("reflection")["analysis"]
    client = None
    if args.command != "report":
        import anthropic
        client = anthropic.Anthropic()
    for name, setup in Setup.load_all().items():
        if args.setups and name not in args.setups:
            continue
        agent = ReflectionAgent(setup, writers[setup.writer], checks)
        contexts = build_contexts(agent, split, args.questions or sample["questions"], sample["seed"])
        paths = stage_paths(agent)
        critiques, revisions = read_stage(paths, "critic"), read_stage(paths, "revise")
        if args.command == "estimate":
            e = estimate_setup(client, agent, contexts, costs, critiques, revisions)
            print(f"{name:28} critic {e['critic_requests']} requests ≈ ${e['critic_dollars']:.2f}   "
                  f"revise {e['revise_requests']} requests ≈ ${e['revise_dollars']:.2f}")
        elif args.command == "run":
            run_setup(client, agent, contexts, costs, paths, args.max_dollars)
            print(f"{name}: done; run `report` to score it")
        else:
            r = report(split, contexts, critiques, revisions)
            a = RevisionAnalysis(analysis["bootstrap_resamples"], analysis["seed"]).analyze(
                ReflectionAgent.revisions(contexts, ReflectionAgent.final_answers(contexts, critiques, revisions)))
            ran = has_run(agent, contexts, critiques, revisions)
            status = (f"critic: {r['critiques']} reviews ({r['critic_revise']} revise, {r['critic_errors']} errors)  "
                      f"revisions: {r['revisions']} ({r['revision_failures']} failed), {r['changed']} changed: answer-F1 "
                      f"better {r['better']}, worse {r['worse']}") if ran else "not run yet"
            print(f"{name:28} answer-F1 {r['draft_answer_f1']:.3f} -> {r['answer_f1']:.3f}  evidence-F1 "
                  f"{r['evidence_f1']:.3f}  unanswerable {r['predicted_unanswerable']:.1%}  ${r['cost']:.2f}\n{'':30}{status}")
            if ran:
                kinds = "  ".join(f"{k.replace('_', ' ')} {v['n']} ({v['mean_delta']:+.3f})" for k, v in a["kinds"].items())
                print(f"{'':30}change {a['delta']:+.3f}, 95% CI [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]  {kinds}")


if __name__ == "__main__":
    main()
