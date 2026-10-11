"""One round of reflection on the M1 drafts: code checks and an optional critic flag problems, and the writer revises
the flagged drafts once. Each setup's files are named with a fingerprint of everything that shapes its requests."""

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reflective_qa.answering import RunConfig, build_request, load_runs
from reflective_qa.baselines import RUNS_DIR, fingerprint as run_fingerprint, read_answers, run_paths, sample_questions
from reflective_qa.checks import CONFIG, Finding, load_max_words, run_checks
from reflective_qa.critic import CriticConfig, build_critic_request
from reflective_qa.data import Paper, Paragraph, Question, text_only_questions
from reflective_qa.retrieval import PaperRetriever

REVISE_INSTRUCTION = ("A reviewer found these problems with your draft. Fix them and answer again in the same format. "
                      "If a problem is wrong, keep that part of the draft.")


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
