"""One round of reflection on the M1 drafts: code checks and an optional critic flag problems, and the writer revises
the flagged drafts once. Each setup's files are named with a fingerprint of everything that shapes its requests."""

import argparse
import hashlib
import json
import statistics
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from reflective_qa.answering import CONFIG as ANSWERING, RunConfig, build_request, load_runs, parse_message
from reflective_qa.baselines import (RUNS_DIR, Costs, count_input_tokens, fingerprint as run_fingerprint, load_costs,
                                     read_answers, run_paths, sample_questions, submit_and_collect)
from reflective_qa.checks import CONFIG, Finding, load_max_words, run_checks
from reflective_qa.critic import CriticConfig, build_critic_request, parse_critique
from reflective_qa.data import Paper, Paragraph, Question, load_split, text_only_questions
from reflective_qa.retrieval import PaperRetriever
from reflective_qa.scoring import Prediction, answer_f1, score

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


def build_contexts(setup: Setup, papers: list[Paper], n: int, seed: int, max_words: int,
                   runs_dir: Path = RUNS_DIR) -> dict[str, Context]:
    writer = load_runs()[setup.writer]
    drafts = {row["question_id"]: row for row in read_answers(run_paths(writer, runs_dir)[0])}
    questions = sample_questions(text_only_questions(papers), n, seed)
    if missing := [q.id for q in questions if q.id not in drafts]:
        raise ValueError(f"{setup.writer} has no draft for {len(missing)} of {len(questions)} sampled questions; "
                         "run its M1 baseline first")
    by_id = {paper.id: paper for paper in papers}
    retrievers: dict[str, PaperRetriever] = {}
    contexts = {}
    for question in questions:
        draft = drafts[question.id]
        paper = by_id[question.paper_id]
        if paper.id not in retrievers:
            retrievers[paper.id] = PaperRetriever(paper)
        passages = retrievers[paper.id].retrieve(question.text, writer.k)
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
                "model": c.model, "cost": costs.dollars(setup.critic.model, c.input_tokens, c.output_tokens)}
    return record


def revision_record(writer: RunConfig, costs: Costs) -> Callable[[str, Any], dict]:
    def record(qid: str, message: Any) -> dict:
        a = parse_message(qid, message)
        return {"question_id": qid, "predicted_answer": a.answer, "predicted_evidence": list(a.evidence_ids),
                "status": a.status, "model": a.model, "output_tokens": a.output_tokens,
                "cost": costs.dollars(writer.model, a.input_tokens, a.output_tokens)}
    return record


def revise_requests(setup: Setup, writer: RunConfig, contexts: dict[str, Context], critiques: dict[str, dict]) -> dict[str, dict]:
    """Only drafts with something to fix are revised."""
    return {qid: revise_request(writer, c, found) for qid, c in contexts.items()
            if (found := problems(setup, c, critiques.get(qid)))}


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


def report(papers: list[Paper], contexts: dict[str, Context], final: dict[str, dict]) -> dict:
    """The setup's final answers scored like M1, plus how reflection changed answers question by question."""
    to_prediction = lambda row: Prediction(row["predicted_answer"], tuple(row["predicted_evidence"]))
    ids = set(contexts)
    result = score(papers, {qid: to_prediction(row) for qid, row in final.items()}, ids)
    drafts = score(papers, {qid: to_prediction(c.draft) for qid, c in contexts.items()}, ids)
    deltas = [answer_f1(c.question, final[qid]["predicted_answer"])[0] - answer_f1(c.question, c.draft["predicted_answer"])[0]
              for qid, c in contexts.items() if final[qid]["revised"]]
    return result | {"draft_answer_f1": drafts["answer_f1"], "revised": len(deltas),
                     "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas),
                     "cost": sum(row["cost"] for row in final.values())}


def stage_estimate(client: Any, model: str, requests: dict[str, dict], costs: Costs, output_tokens: float) -> float:
    return costs.dollars(model, count_input_tokens(client, requests) * len(requests), output_tokens * len(requests)) if requests else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "run", "report"])
    parser.add_argument("setups", nargs="*", help="setup names from configs/reflection.toml; default: all")
    parser.add_argument("--dev", type=Path, default=RUNS_DIR.parent / "raw" / "qasper-dev-v0.3.json")
    parser.add_argument("--questions", type=int, help="size of the M1 question sample; default: the config's")
    parser.add_argument("--max-dollars", type=float, help="required for run: per stage, refuse if the estimate is higher")
    args = parser.parse_args()
    sample = tomllib.loads(ANSWERING.read_text())["sample"]
    papers, costs, runs, max_words = load_split(args.dev), load_costs(), load_runs(), load_max_words()
    client = None
    if args.command != "report":
        import anthropic
        client = anthropic.Anthropic()
    for name, setup in load_setups().items():
        if args.setups and name not in args.setups:
            continue
        writer = runs[setup.writer]
        contexts = build_contexts(setup, papers, args.questions or sample["questions"], sample["seed"], max_words)
        paths = stage_paths(setup, writer, max_words)
        critiques = {r["question_id"]: r for r in read_answers(paths["critic"][0])}
        revisions = {r["question_id"]: r for r in read_answers(paths["revise"][0])}
        if args.command == "report":
            r = report(papers, contexts, final_answers(contexts, critiques, revisions))
            print(f"{name:28} answer-F1 {r['draft_answer_f1']:.3f} -> {r['answer_f1']:.3f}  evidence-F1 {r['evidence_f1']:.3f}  "
                  f"unanswerable {r['predicted_unanswerable']:.1%}  revised {r['revised']} (better {r['better']}, worse "
                  f"{r['worse']})  ${r['cost']:.2f}")
            continue
        draft_output = statistics.mean(c.draft["output_tokens"] for c in contexts.values())
        pending_critic = {q: p for q, p in critic_requests(setup, contexts).items() if q not in critiques}
        critic_cost = stage_estimate(client, setup.critic.model, pending_critic, costs, costs.assumed_output_tokens) if setup.critic else 0.0
        if args.command == "estimate":
            # Upper bound for the revise stage: every draft revised, unless the critic has already run.
            found = lambda c: problems(setup, c, critiques.get(c.question.id)) or ([Finding("wrong_or_incomplete", "x")] if setup.critic else [])
            upper = {q: revise_request(writer, c, f) for q, c in contexts.items() if q not in revisions and (f := found(c))}
            revise_cost = stage_estimate(client, writer.model, upper, costs, draft_output)
            print(f"{name:28} critic {len(pending_critic)} requests ≈ ${critic_cost:.2f}  revise up to {len(upper)} ≈ ${revise_cost:.2f}")
            continue
        if args.max_dollars is None:
            raise SystemExit("run needs --max-dollars")
        if setup.critic:
            if critic_cost > args.max_dollars and not paths["critic"][1].exists():
                raise SystemExit(f"{name}: critic estimate ${critic_cost:.2f} is over --max-dollars {args.max_dollars}")
            submit_and_collect(client, critic_requests(setup, contexts), *paths["critic"], critique_record(setup, costs))
            critiques = {r["question_id"]: r for r in read_answers(paths["critic"][0])}
        pending_revise = {q: p for q, p in revise_requests(setup, writer, contexts, critiques).items() if q not in revisions}
        revise_cost = stage_estimate(client, writer.model, pending_revise, costs, draft_output)
        if revise_cost > args.max_dollars and not paths["revise"][1].exists():
            raise SystemExit(f"{name}: revise estimate ${revise_cost:.2f} is over --max-dollars {args.max_dollars}")
        submit_and_collect(client, revise_requests(setup, writer, contexts, critiques), *paths["revise"], revision_record(writer, costs))
        print(f"{name}: done; run `report` to score it")


if __name__ == "__main__":
    main()
