"""Run a writer config over the text-only dev questions with the Batch API, caching every answer on disk.

A rerun never pays twice: answered questions are skipped, and a submitted batch's ID is saved before waiting,
so an interrupted run resumes the same batch instead of submitting a new one.
"""

import argparse
import json
import statistics
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflective_qa.answering import CONFIG, Answer, RunConfig, build_request, load_runs, parse_message
from reflective_qa.data import Paper, load_split, text_only_questions
from reflective_qa.retrieval import PaperRetriever

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


def build_requests(run: RunConfig, papers: list[Paper], limit: int | None = None) -> dict[str, dict[str, Any]]:
    """Request parameters keyed by question ID, in dataset order."""
    by_id = {paper.id: paper for paper in papers}
    retrievers: dict[str, PaperRetriever] = {}
    requests = {}
    for question in text_only_questions(papers)[:limit]:
        paper = by_id[question.paper_id]
        retriever = retrievers.setdefault(paper.id, PaperRetriever(paper))
        passages = retriever.retrieve(question.text, run.k) if run.k else []
        requests[question.id] = build_request(run, paper, question, passages)
    return requests


def read_answers(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_batch(client: Any, run: RunConfig, requests: dict[str, dict[str, Any]], out_dir: Path = RUNS_DIR,
              poll_seconds: float = 30) -> Path:
    """Answers go to <run>.jsonl in the shape `scoring.load_predictions` reads. Only succeeded results are
    written, so errored or expired requests are retried on the next run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path, batch_path = out_dir / f"{run.name}.jsonl", out_dir / f"{run.name}.batch"
    done = {row["question_id"] for row in read_answers(answers_path)}
    if batch_path.exists():
        batch_id = batch_path.read_text().strip()
    else:
        pending = [{"custom_id": qid, "params": params} for qid, params in requests.items() if qid not in done]
        if not pending:
            return answers_path
        batch_id = client.messages.batches.create(requests=pending).id
        batch_path.write_text(batch_id)
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll_seconds)
    with answers_path.open("a") as out:
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded" or result.custom_id in done:
                continue
            answer = parse_message(result.custom_id, result.result.message)
            out.write(json.dumps(_record(answer, run)) + "\n")
            done.add(result.custom_id)
    batch_path.unlink()
    return answers_path


def _record(answer: Answer, run: RunConfig) -> dict:
    return {"question_id": answer.question_id, "predicted_answer": answer.answer,
            "predicted_evidence": list(answer.evidence_ids), "status": answer.status, "model": run.model,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens}


def estimate(client: Any, run: RunConfig, requests: dict[str, dict[str, Any]], costs: Costs, sample: int = 50,
             out_dir: Path = RUNS_DIR) -> dict:
    """Batch cost of the whole run. Input tokens are counted (free) on an evenly spaced sample; output tokens are
    this run's measured mean if it has answers, else the configured assumption."""
    ids = list(requests)
    sampled = ids[:: max(1, len(ids) // sample)][:sample]
    counted = [client.messages.count_tokens(**{k: v for k, v in requests[qid].items() if k != "max_tokens"}).input_tokens
               for qid in sampled]
    answered = read_answers(out_dir / f"{run.name}.jsonl")
    output = statistics.mean(r["output_tokens"] for r in answered) if answered else costs.assumed_output_tokens
    input_mean = statistics.mean(counted)
    return {"run": run.name, "questions": len(ids), "input_tokens": input_mean, "output_tokens": output,
            "output_measured": bool(answered),
            "dollars": costs.dollars(run.model, input_mean * len(ids), output * len(ids))}


def main() -> None:
    import anthropic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "run"])
    parser.add_argument("runs", nargs="*", help="run names from configs/answering.toml; default: all")
    parser.add_argument("--dev", type=Path, default=RUNS_DIR.parent / "raw" / "qasper-dev-v0.3.json")
    parser.add_argument("--limit", type=int, help="only the first N text-only questions")
    args = parser.parse_args()
    client, papers, costs, runs = anthropic.Anthropic(), load_split(args.dev), load_costs(), load_runs()
    for name in args.runs or runs:
        run = runs[name]
        requests = build_requests(run, papers, args.limit)
        if args.command == "estimate":
            e = estimate(client, run, requests, costs)
            print(f"{e['run']:20} {e['questions']:>4} questions  ~{e['input_tokens']:,.0f} input tokens  "
                  f"{e['output_tokens']:,.0f} output tokens ({'measured' if e['output_measured'] else 'assumed'})  "
                  f"≈ ${e['dollars']:.2f}")
        else:
            rows = read_answers(run_batch(client, run, requests))
            spent = costs.dollars(run.model, sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows))
            statuses = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
            missing = len(set(requests) - {r["question_id"] for r in rows})
            print(f"{run.name}: {len(rows)} answers {statuses}, {missing} requested but missing, ${spent:.2f} so far")


if __name__ == "__main__":
    main()
