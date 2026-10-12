"""One round of reflection on the M1 drafts: code checks and an optional critic flag problems, and the writer revises
the flagged drafts once. Each setup's files are named with a fingerprint of everything that shapes its requests."""

import argparse
import hashlib
import json
import random
import statistics
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from reflective_qa.answering import CONFIG as ANSWERING, RunConfig, build_request, load_runs, parse_message
from reflective_qa.baselines import (RUNS_DIR, Costs, answer_record, count_input_tokens, fingerprint as run_fingerprint,
                                     load_costs, read_answers, run_paths, submit_and_collect)
from reflective_qa.checks import CONFIG, Finding, load_max_words, run_checks
from reflective_qa.critic import CriticConfig, build_critic_request, parse_critique
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.records import Paper, Paragraph, Question
from reflective_qa.paths import DEV_SPLIT
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.scoring import Prediction, answer_f1, is_unanswerable, score

# The answer rules are repeated because the 20-question pilot showed revisions drifting long and hedged
# ("Yes, partly: ..."), which the writer's system prompt alone did not prevent.
REVISE_INSTRUCTION = ("A reviewer found these problems with your draft. Fix them and answer again in the same format. "
                      "Keep the same rules: the shortest phrase that answers the question, Yes or No for a yes/no "
                      "question, and Unanswerable only if the passages do not contain the answer. If a problem is "
                      "wrong, keep that part of the draft.")


@dataclass(frozen=True)
class Setup:
    name: str
    writer: str
    """A run in configs/answering.toml: its answers are the drafts, and its model writes the revision."""
    critic: CriticConfig | None
    """None means code feedback only: the checks' findings go straight to the writer."""


@dataclass(frozen=True)
class Context:
    """Everything one question's reflection needs: what the writer saw, what it answered, what the checks found."""
    paper: Paper
    question: Question
    passages: list[Paragraph]
    draft: dict
    findings: list[Finding]


def load_setups(path: Path = CONFIG) -> dict[str, Setup]:
    config = tomllib.loads(path.read_text())
    setups = {}
    for name, s in config["setups"].items():
        critic = None if s["critic"] == "none" else CriticConfig(s["critic"], config["critic"]["effort"],
                                                                config["critic"]["max_tokens"], s["code_feedback"])
        setups[name] = Setup(name, s["writer"], critic)
    return setups


def _probe(max_words: int) -> Context:
    """A fixed question whose draft trips several checks, so the fingerprint sees real finding texts."""
    paragraphs = [Paragraph("s0p0", "Probe", "A probe paragraph."), Paragraph("s1p0", "Probe", "Another probe paragraph.")]
    question = Question("probe", "probe", "Do they use a probe?", ())
    answer = " ".join(["word"] * 60)  # fixed, so the draft text never depends on the word limit
    findings = run_checks(question.text, answer, ("s9p9",), {p.id for p in paragraphs}, max_words)
    draft = {"question_id": "probe", "predicted_answer": answer, "predicted_evidence": ["s9p9"], "status": "ok"}
    return Context(Paper("probe", "Probe paper", "", tuple(paragraphs), (question,)), question, paragraphs, draft, findings)


def fingerprint(setup: Setup, writer: RunConfig, max_words: int) -> str:
    """A hash of the actual critic and revise requests for a fixed probe question, so any change to a prompt, schema,
    model setting, finding text or check behavior starts new files. The word limit counts only where findings are
    shown: a critic that never sees them keeps its paid reviews when the limit changes."""
    probe = _probe(max_words)
    critique = {"verdict": "revise", "problems": [{"check": "unsupported", "detail": "probe"}]}
    parts = {"writer": run_fingerprint(writer), "critic_requests": critic_requests(setup, {"probe": probe}),
             "revise_request": revise_request(writer, probe, problems(setup, probe, critique))}
    if setup.critic is None or setup.critic.code_feedback:
        parts["max_words"] = max_words  # it decides which drafts get findings
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def stage_paths(setup: Setup, writer: RunConfig, max_words: int, out_dir: Path = RUNS_DIR) -> dict[str, tuple[Path, Path]]:
    """(answers, batch) file pairs for the critic and revise stages."""
    stem = f"reflect-{setup.name}-{fingerprint(setup, writer, max_words)}"
    return {stage: (out_dir / f"{stem}.{stage}.jsonl", out_dir / f"{stem}.{stage}.batch") for stage in ("critic", "revise")}


def read_stage(paths: dict[str, tuple[Path, Path]], stage: str) -> dict[str, dict]:
    """One stage's results on disk, keyed by question ID."""
    return {r["question_id"]: r for r in read_answers(paths[stage][0])}


def build_contexts(setup: Setup, split: QasperSplit, n: int, seed: int, max_words: int,
                   runs_dir: Path = RUNS_DIR) -> dict[str, Context]:
    writer = load_runs()[setup.writer]
    drafts = {row["question_id"]: row for row in read_answers(run_paths(writer, runs_dir)[0])}
    questions = split.sample(n, seed)
    if missing := [q.id for q in questions if q.id not in drafts]:
        raise ValueError(f"{setup.writer} has no draft for {len(missing)} of {len(questions)} sampled questions; "
                         "run its M1 baseline first")
    retriever = Retriever(split)
    contexts = {}
    for question in questions:
        draft = drafts[question.id]
        paper = split.paper(question.paper_id)
        passages = retriever.retrieve(question, writer.k)
        findings = run_checks(question.text, draft["predicted_answer"], tuple(draft["predicted_evidence"]),
                              {p.id for p in passages}, max_words)
        contexts[question.id] = Context(paper, question, passages, draft, findings)
    return contexts


def critic_requests(setup: Setup, contexts: dict[str, Context]) -> dict[str, dict]:
    if setup.critic is None:
        return {}
    return {qid: build_critic_request(setup.critic, c.paper, c.question, c.passages, c.draft["predicted_answer"],
                                      tuple(c.draft["predicted_evidence"]), c.findings) for qid, c in contexts.items()}


def problems(setup: Setup, context: Context, critique: dict | None) -> list[Finding]:
    """What the writer is asked to fix: the critic's problems, or with no critic, the checks' findings."""
    if setup.critic is None:
        return context.findings
    if critique is None or critique["verdict"] != "revise":
        return []
    return [Finding(p["check"], p["detail"]) for p in critique["problems"]]


def revise_request(writer: RunConfig, context: Context, found: list[Finding]) -> dict[str, Any]:
    """The writer's original request, plus its draft and the problems, in one user turn."""
    request = build_request(writer, context.paper, context.question, context.passages)
    draft = json.dumps({"answer": context.draft["predicted_answer"], "evidence": context.draft["predicted_evidence"]})
    feedback = "\n".join(f"- {f.check}: {f.detail}" for f in found)
    content = f"{request['messages'][0]['content']}\n\nYour draft:\n{draft}\n\n{REVISE_INSTRUCTION}\n{feedback}"
    return request | {"messages": [{"role": "user", "content": content}]}


def critique_record(setup: Setup, costs: Costs) -> Callable[[str, Any], dict]:
    def record(qid: str, message: Any) -> dict:
        c = parse_critique(qid, message)
        return {"question_id": qid, "verdict": c.verdict, "problems": [asdict(f) for f in c.problems], "status": c.status,
                "model": c.model, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "cost": costs.dollars(setup.critic.model, c.input_tokens, c.output_tokens)}
    return record


def revise_requests(setup: Setup, writer: RunConfig, contexts: dict[str, Context],
                    critiques: dict[str, dict]) -> dict[str, dict]:
    """Only drafts with something to fix are revised."""
    return {qid: revise_request(writer, c, found) for qid, c in contexts.items()
            if (found := problems(setup, c, critiques.get(qid)))}


def has_run(setup: Setup, writer: RunConfig, contexts: dict[str, Context], critiques: dict[str, dict],
            revisions: dict[str, dict]) -> bool:
    """Whether the setup has results for these questions. Code feedback that flags no draft has nothing to run."""
    return bool(set(contexts) & (set(critiques) | set(revisions))) or (
        setup.critic is None and not revise_requests(setup, writer, contexts, {}))


def final_answers(contexts: dict[str, Context], critiques: dict[str, dict], revisions: dict[str, dict]) -> dict[str, dict]:
    """The revision where one succeeded, else the draft; with the cost of reflecting, not of the draft."""
    final = {}
    for qid, c in contexts.items():
        revised = revisions.get(qid)
        use = revised if revised and revised["status"] == "ok" else c.draft
        cost = (critiques.get(qid) or {}).get("cost", 0.0) + (revised or {}).get("cost", 0.0)
        final[qid] = {"question_id": qid, "predicted_answer": use["predicted_answer"],
                      "predicted_evidence": use["predicted_evidence"], "revised": use is revised, "cost": cost}
    return final


def report(split: QasperSplit, contexts: dict[str, Context], critiques: dict[str, dict],
           revisions: dict[str, dict]) -> dict:
    """Final answers scored like M1, plus what reflection did. better and worse compare answer-F1 per question,
    over the answers a revision actually changed."""
    final = final_answers(contexts, critiques, revisions)
    to_prediction = lambda row: Prediction(row["predicted_answer"], tuple(row["predicted_evidence"]))
    ids = set(contexts)
    result = score(split, {qid: to_prediction(row) for qid, row in final.items()}, ids)
    drafts = score(split, {qid: to_prediction(c.draft) for qid, c in contexts.items()}, ids)
    changed = [qid for qid, c in contexts.items() if final[qid]["revised"]
               and final[qid]["predicted_answer"] != c.draft["predicted_answer"]]
    delta = {qid: answer_f1(contexts[qid].question, final[qid]["predicted_answer"])[0]
             - answer_f1(contexts[qid].question, contexts[qid].draft["predicted_answer"])[0] for qid in changed}
    verdicts = [r["verdict"] for r in critiques.values() if r["question_id"] in ids]
    return result | {"draft_answer_f1": drafts["answer_f1"], "critiques": len(verdicts),
                     "critic_revise": verdicts.count("revise"), "critic_errors": verdicts.count("error"),
                     "revisions": sum(qid in ids for qid in revisions), "revision_failures": sum(
                         r["status"] != "ok" for qid, r in revisions.items() if qid in ids),
                     "changed": len(changed), "better": sum(d > 0 for d in delta.values()),
                     "worse": sum(d < 0 for d in delta.values()), "cost": sum(row["cost"] for row in final.values())}


def revision_kind(before: str, after: str) -> str:
    """How an answer changed: an abstention turned into an answer, an answer turned into an abstention, or a rewording."""
    return ("abstain_to_answer" if is_unanswerable(before) and not is_unanswerable(after) else
            "answer_to_abstain" if is_unanswerable(after) and not is_unanswerable(before) else "reworded")


def analyze(contexts: dict[str, Context], final: dict[str, dict], resamples: int, seed: int) -> dict:
    """How reflection moved answer-F1: the mean change per question with a paired-bootstrap 95% interval, and the
    changed answers split by kind: an abstention turned into an answer, an answer turned into an abstention, or a
    rewording."""
    deltas = {q: answer_f1(c.question, final[q]["predicted_answer"])[0] - answer_f1(c.question, c.draft["predicted_answer"])[0]
              for q, c in contexts.items()}
    rng, values = random.Random(seed), list(deltas.values())
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    kinds: dict[str, list[float]] = {"abstain_to_answer": [], "answer_to_abstain": [], "reworded": []}
    for q, c in contexts.items():
        before, after = c.draft["predicted_answer"], final[q]["predicted_answer"]
        if before == after:
            continue
        kinds[revision_kind(before, after)].append(deltas[q])
    return {"delta": statistics.mean(values), "ci": (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]),
            "kinds": {k: {"n": len(v), "mean_delta": statistics.mean(v) if v else 0.0} for k, v in kinds.items()}}


def _mean_output(rows: list[dict], default: float) -> float:
    return statistics.mean(r["output_tokens"] for r in rows) if rows else default


def _dollars(client: Any, model: str, requests: dict[str, dict], costs: Costs, output_tokens: float) -> float:
    if not requests:
        return 0.0
    return costs.dollars(model, count_input_tokens(client, requests) * len(requests), output_tokens * len(requests))


def estimate_setup(client: Any, setup: Setup, writer: RunConfig, contexts: dict[str, Context], costs: Costs,
                   critiques: dict[str, dict], revisions: dict[str, dict]) -> dict:
    """Batch cost of what is left to run. Output tokens are measured once a stage has results, else assumed (critic)
    or taken from the drafts (revisions). Before the critic has run, every draft is assumed to need a revision."""
    pending_critic = {q: p for q, p in critic_requests(setup, contexts).items() if q not in critiques}
    stand_in = [Finding("wrong_or_incomplete", "stand-in for an unknown critique")]
    found = {q: problems(setup, c, critiques.get(q)) if (setup.critic is None or q in critiques) else stand_in
             for q, c in contexts.items()}
    pending_revise = {q: revise_request(writer, contexts[q], f) for q, f in found.items() if f and q not in revisions}
    draft_output = _mean_output([c.draft for c in contexts.values()], costs.assumed_output_tokens)
    critic_dollars = (_dollars(client, setup.critic.model, pending_critic, costs,
                               _mean_output(list(critiques.values()), costs.assumed_output_tokens)) if setup.critic else 0.0)
    revise_dollars = _dollars(client, writer.model, pending_revise, costs, _mean_output(list(revisions.values()), draft_output))
    return {"critic_requests": len(pending_critic), "critic_dollars": critic_dollars,
            "revise_requests": len(pending_revise), "revise_dollars": revise_dollars}


def run_setup(client: Any, setup: Setup, writer: RunConfig, contexts: dict[str, Context], costs: Costs,
              paths: dict[str, tuple[Path, Path]], max_dollars: float, poll_seconds: float = 30) -> None:
    """Critic stage, then revise stage. Each is refused when its estimate is over max_dollars, unless a batch for it
    is already paid for and waiting to be collected."""
    read = lambda stage: read_stage(paths, stage)
    if setup.critic:
        e = estimate_setup(client, setup, writer, contexts, costs, read("critic"), read("revise"))
        if e["critic_dollars"] > max_dollars and not paths["critic"][1].exists():
            raise SystemExit(f"{setup.name}: critic estimate ${e['critic_dollars']:.2f} is over --max-dollars {max_dollars}")
        submit_and_collect(client, critic_requests(setup, contexts), *paths["critic"], critique_record(setup, costs),
                           poll_seconds)
    critiques = read("critic")
    e = estimate_setup(client, setup, writer, contexts, costs, critiques, read("revise"))
    if e["revise_dollars"] > max_dollars and not paths["revise"][1].exists():
        raise SystemExit(f"{setup.name}: revise estimate ${e['revise_dollars']:.2f} is over --max-dollars {max_dollars}")
    record = lambda qid, message: answer_record(parse_message(qid, message), writer, costs)
    submit_and_collect(client, revise_requests(setup, writer, contexts, critiques), *paths["revise"], record, poll_seconds)


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
    sample = tomllib.loads(ANSWERING.read_text())["sample"]
    split, costs, runs, max_words = QasperSplit.load(args.dev), load_costs(), load_runs(), load_max_words()
    analysis = tomllib.loads(CONFIG.read_text())["analysis"]
    client = None
    if args.command != "report":
        import anthropic
        client = anthropic.Anthropic()
    for name, setup in load_setups().items():
        if args.setups and name not in args.setups:
            continue
        writer = runs[setup.writer]
        contexts = build_contexts(setup, split, args.questions or sample["questions"], sample["seed"], max_words)
        paths = stage_paths(setup, writer, max_words)
        critiques, revisions = read_stage(paths, "critic"), read_stage(paths, "revise")
        if args.command == "estimate":
            e = estimate_setup(client, setup, writer, contexts, costs, critiques, revisions)
            print(f"{name:28} critic {e['critic_requests']} requests ≈ ${e['critic_dollars']:.2f}   "
                  f"revise {e['revise_requests']} requests ≈ ${e['revise_dollars']:.2f}")
        elif args.command == "run":
            run_setup(client, setup, writer, contexts, costs, paths, args.max_dollars)
            print(f"{name}: done; run `report` to score it")
        else:
            r = report(split, contexts, critiques, revisions)
            a = analyze(contexts, final_answers(contexts, critiques, revisions), analysis["bootstrap_resamples"], analysis["seed"])
            ran = has_run(setup, writer, contexts, critiques, revisions)
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
