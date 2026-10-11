"""Run a writer config over a fixed sample of text-only dev questions with the Batch API, caching every answer.

Money safety:
- A rerun never pays twice. Answered questions are skipped, and a submitted batch's ID is saved before waiting,
  so an interrupted run resumes that batch.
- Answers and batch IDs are filed under a fingerprint of the run config, prompt and schema. Changing any of them
  starts a new file instead of mixing answers from two setups.
- `run` refuses to submit when the estimated cost of the unanswered questions exceeds --max-dollars.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from reflective_qa.answering import (ANSWER_SCHEMA, CONFIG, SYSTEM, Answer, RunConfig, build_request, load_runs,
                                     parse_message)
from reflective_qa.data import Paper, Question, load_split, text_only_questions
from reflective_qa.retrieval import PaperRetriever
from reflective_qa.scoring import load_predictions, score

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"


@dataclass(frozen=True)
class Costs:
    prices: dict[str, dict[str, float]]
    batch_discount: float
    assumed_output_tokens: int

    def dollars(self, model: str, input_tokens: float, output_tokens: float) -> float:
        price = self.prices[model]
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6 * self.batch_discount


def load_costs(path: Path = CONFIG) -> Costs:
    return Costs(**tomllib.loads(path.read_text())["costs"])


def fingerprint(run: RunConfig) -> str:
    setup = {"run": {k: v for k, v in asdict(run).items() if k != "name"}, "system": SYSTEM, "schema": ANSWER_SCHEMA}
    return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:10]


def run_paths(run: RunConfig, out_dir: Path) -> tuple[Path, Path]:
    stem = f"{run.name}-{fingerprint(run)}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.batch"


def sample_questions(questions: list[Question], n: int | None, seed: int) -> list[Question]:
    """A seeded shuffle, cut to n. Smaller samples are prefixes of larger ones, so a pilot's answers are reused."""
    order = sorted(questions, key=lambda q: q.id)
    random.Random(seed).shuffle(order)
    return order[:n]


def build_requests(run: RunConfig, papers: list[Paper], n: int | None = None, seed: int = 0) -> dict[str, dict]:
    """Request parameters keyed by question ID, for a seeded sample of n text-only questions (all if n is None)."""
    by_id = {paper.id: paper for paper in papers}
    retrievers: dict[str, PaperRetriever] = {}
    requests = {}
    for question in sample_questions(text_only_questions(papers), n, seed):
        paper = by_id[question.paper_id]
        if paper.id not in retrievers:
            retrievers[paper.id] = PaperRetriever(paper)
        passages = retrievers[paper.id].retrieve(question.text, run.k) if run.k else []
        requests[question.id] = build_request(run, paper, question, passages)
    return requests


def read_answers(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def submit_and_collect(client: Any, requests: dict[str, dict], answers_path: Path, batch_path: Path,
                       record: Callable[[str, Any], dict], poll_seconds: float = 30) -> Path:
    """Batch the requests not yet in `answers_path`, wait, and append `record(custom_id, message)` for each succeeded
    result. Only succeeded results are written, so errored or expired requests are retried on the next call."""
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["question_id"] for row in read_answers(answers_path)}
    if batch_path.exists():
        batch_id = batch_path.read_text().strip()
    else:
        pending = [{"custom_id": qid, "params": params} for qid, params in requests.items() if qid not in done]
        if not pending:
            return answers_path
        # No SDK retries: if a retried create's first response were lost, a second paid batch would be submitted.
        # After a failed create, check client.messages.batches.list() before running again.
        batch_id = client.with_options(max_retries=0).messages.batches.create(requests=pending).id
        batch_path.write_text(batch_id)
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll_seconds)
    with answers_path.open("a") as out:
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded" or result.custom_id in done:
                continue
            out.write(json.dumps(record(result.custom_id, result.result.message)) + "\n")
            done.add(result.custom_id)
    batch_path.unlink()
    return answers_path


def run_batch(client: Any, run: RunConfig, requests: dict[str, dict], costs: Costs, out_dir: Path = RUNS_DIR,
              poll_seconds: float = 30) -> Path:
    """Answers go to the run's JSON lines file, in the shape `scoring.load_predictions` reads."""
    answers_path, batch_path = run_paths(run, out_dir)
    return submit_and_collect(client, requests, answers_path, batch_path,
                              lambda qid, message: _record(parse_message(qid, message), run, costs), poll_seconds)


def _record(answer: Answer, run: RunConfig, costs: Costs) -> dict:
    return {"question_id": answer.question_id, "predicted_answer": answer.answer,
            "predicted_evidence": list(answer.evidence_ids), "status": answer.status, "model": answer.model,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
            "cost": costs.dollars(run.model, answer.input_tokens, answer.output_tokens)}


def estimate(client: Any, run: RunConfig, requests: dict[str, dict], costs: Costs, sample: int = 50,
             out_dir: Path = RUNS_DIR) -> dict:
    """Batch cost of answering `requests`. Input tokens are counted (free) on an evenly spaced sample; output
    tokens are this run's measured mean if it has answers, else the configured assumption."""
    ids = list(requests)
    if not ids:
        return {"run": run.name, "questions": 0, "input_tokens": 0, "output_tokens": 0, "output_measured": False,
                "dollars": 0.0}
    sampled = ids[:: max(1, len(ids) // sample)][:sample]
    counted = [client.messages.count_tokens(**{k: v for k, v in requests[qid].items() if k != "max_tokens"}).input_tokens
               for qid in sampled]
    answered = read_answers(run_paths(run, out_dir)[0])
    output = statistics.mean(r["output_tokens"] for r in answered) if answered else costs.assumed_output_tokens
    input_mean = statistics.mean(counted)
    return {"run": run.name, "questions": len(ids), "input_tokens": input_mean, "output_tokens": output,
            "output_measured": bool(answered),
            "dollars": costs.dollars(run.model, input_mean * len(ids), output * len(ids))}


def report(run: RunConfig, requests: dict[str, dict], papers: list[Paper], out_dir: Path = RUNS_DIR) -> dict:
    """Scores a run on its own question sample, so questions outside the sample don't count as zeros."""
    answers_path = run_paths(run, out_dir)[0]
    rows = read_answers(answers_path)
    result = score(papers, load_predictions(answers_path) if rows else {}, set(requests))
    return result | {"run": run.name, "file": answers_path.name, "cost": sum(r["cost"] for r in rows),
                     "output_tokens": statistics.mean(r["output_tokens"] for r in rows) if rows else 0}


def total_spent(out_dir: Path = RUNS_DIR) -> float:
    return sum(row["cost"] for path in out_dir.glob("*.jsonl") for row in read_answers(path))


def main() -> None:
    import anthropic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "run", "report"])
    parser.add_argument("runs", nargs="*", help="run names from configs/answering.toml; default: all")
    parser.add_argument("--dev", type=Path, default=RUNS_DIR.parent / "raw" / "qasper-dev-v0.3.json")
    parser.add_argument("--questions", type=int, help="size of the seeded question sample; default: the config's")
    parser.add_argument("--max-dollars", type=float, help="required for run: refuse if the estimate is higher")
    args = parser.parse_args()
    config = tomllib.loads(CONFIG.read_text())["sample"]
    n = args.questions or config["questions"]
    papers, costs, runs = load_split(args.dev), load_costs(), load_runs()
    client = anthropic.Anthropic() if args.command != "report" else None
    for name in args.runs or runs:
        run = runs[name]
        requests = build_requests(run, papers, n, config["seed"])
        if args.command == "report":
            r = report(run, requests, papers)
            by_type = "  ".join(f"{t} {f1:.3f}" for t, f1 in r["answer_f1_by_type"].items())
            print(f"{r['run']:20} {r['questions'] - r['missing']}/{r['questions']} answered  answer-F1 {r['answer_f1']:.3f}  "
                  f"evidence-F1 {r['evidence_f1']:.3f}  unanswerable {r['predicted_unanswerable']:.1%}  "
                  f"{r['output_tokens']:.0f} output tokens  ${r['cost']:.2f}\n{'':22}{by_type}")
            continue
        answers_path, batch_path = run_paths(run, RUNS_DIR)
        done = {row["question_id"] for row in read_answers(answers_path)}
        pending = {qid: params for qid, params in requests.items() if qid not in done}
        e = estimate(client, run, pending, costs)
        print(f"{run.name:20} {len(requests)} questions, {e['questions']} unanswered  ~{e['input_tokens']:,.0f} input "
              f"tokens  {e['output_tokens']:,.0f} output ({'measured' if e['output_measured'] else 'assumed'})  "
              f"≈ ${e['dollars']:.2f}")
        if args.command == "run":
            if args.max_dollars is None:
                raise SystemExit("run needs --max-dollars")
            if e["dollars"] > args.max_dollars and not batch_path.exists():
                raise SystemExit(f"{run.name}: estimate ${e['dollars']:.2f} is over --max-dollars {args.max_dollars}")
            rows = read_answers(run_batch(client, run, requests, costs))
            statuses = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
            print(f"  -> {answers_path.name}: {len(rows)} answers {statuses}, "
                  f"${sum(r['cost'] for r in rows):.2f} for this run, ${total_spent():.2f} across all runs")


if __name__ == "__main__":
    main()
