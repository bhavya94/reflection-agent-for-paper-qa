"""Run a writer config over a fixed sample of text-only dev questions with the Batch API, caching every answer.

Money safety:
- A rerun never pays twice. Answered questions are skipped, and a submitted batch's ID is saved before waiting,
  so an interrupted run resumes that batch.
- Answers and batch IDs are filed under a fingerprint of the run config, prompt and schema. Changing any of them
  starts a new file instead of mixing answers from two setups.
- `run` refuses to submit when the estimated cost of the unanswered questions exceeds --max-dollars.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.writer import ModelAnswer, Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.paths import DEV_SPLIT, RUNS_DIR, load_config
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.eval.metrics import answer_f1, is_unanswerable, normalize
from reflective_qa.eval.scorer import Scorer, load_predictions


@dataclass(frozen=True)
class Costs:
    prices: dict[str, dict[str, float]]
    batch_discount: float
    assumed_output_tokens: int

    def dollars(self, model: str, input_tokens: float, output_tokens: float) -> float:
        price = self.prices[model]
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6 * self.batch_discount


def load_costs() -> Costs:
    return Costs(**load_config("answering")["costs"])


def run_paths(writer: Writer, out_dir: Path) -> tuple[Path, Path]:
    stem = f"{writer.name}-{writer.fingerprint}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.batch"


def build_requests(writer: Writer, split: QasperSplit, n: int | None = None, seed: int = 0) -> dict[str, dict]:
    """Request parameters keyed by question ID, for a seeded sample of n text-only questions (all if n is None)."""
    retriever = Retriever(split)
    return {q.id: writer.request(split.paper(q.paper_id), q, retriever.retrieve(q, writer.k)) for q in split.sample(n, seed)}


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


def run_batch(client: Any, writer: Writer, requests: dict[str, dict], costs: Costs, out_dir: Path = RUNS_DIR,
              poll_seconds: float = 30) -> Path:
    """Answers go to the run's JSON lines file, in the shape `load_predictions` reads."""
    answers_path, batch_path = run_paths(writer, out_dir)
    return submit_and_collect(client, requests, answers_path, batch_path,
                              lambda qid, message: answer_record(Writer.parse(qid, message), writer, costs), poll_seconds)


def answer_record(answer: ModelAnswer, writer: Writer, costs: Costs) -> dict:
    return {"question_id": answer.question_id, "predicted_answer": answer.answer,
            "predicted_evidence": list(answer.evidence_ids), "status": answer.status, "model": answer.model,
            "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
            "cost": costs.dollars(writer.model, answer.input_tokens, answer.output_tokens)}


def count_input_tokens(client: Any, requests: dict[str, dict], sample: int = 50) -> float:
    """Mean input tokens over an evenly spaced sample of the requests, counted with the free count_tokens endpoint."""
    ids = list(requests)
    sampled = ids[:: max(1, len(ids) // sample)][:sample]
    return statistics.mean(client.messages.count_tokens(**{k: v for k, v in requests[qid].items() if k != "max_tokens"})
                           .input_tokens for qid in sampled)


def estimate(client: Any, writer: Writer, requests: dict[str, dict], costs: Costs, sample: int = 50,
             out_dir: Path = RUNS_DIR) -> dict:
    """Batch cost of answering `requests`. Input tokens are counted (free) on an evenly spaced sample; output
    tokens are this run's measured mean if it has answers, else the configured assumption."""
    ids = list(requests)
    if not ids:
        return {"run": writer.name, "questions": 0, "input_tokens": 0, "output_tokens": 0, "output_measured": False,
                "dollars": 0.0}
    answered = read_answers(run_paths(writer, out_dir)[0])
    output = statistics.mean(r["output_tokens"] for r in answered) if answered else costs.assumed_output_tokens
    input_mean = count_input_tokens(client, requests, sample)
    return {"run": writer.name, "questions": len(ids), "input_tokens": input_mean, "output_tokens": output,
            "output_measured": bool(answered),
            "dollars": costs.dollars(writer.model, input_mean * len(ids), output * len(ids))}


def report(writer: Writer, requests: dict[str, dict], split: QasperSplit, out_dir: Path = RUNS_DIR) -> dict:
    """Scores a run on its own question sample, so questions outside the sample don't count as zeros."""
    answers_path = run_paths(writer, out_dir)[0]
    rows = read_answers(answers_path)
    result = Scorer(split).score(load_predictions(answers_path) if rows else {}, set(requests))
    return result | {"run": writer.name, "file": answers_path.name, "cost": sum(r["cost"] for r in rows),
                     "output_tokens": statistics.mean(r["output_tokens"] for r in rows) if rows else 0}


def check_findings(writer: Writer, split: QasperSplit, checks: CodeChecks, n: int | None, seed: int,
                   out_dir: Path = RUNS_DIR) -> dict:
    """How often each code check fires on a writer's drafts. The F1 comparison covers content answers only (not
    Unanswerable, Yes or No), since too_long can't flag those; it describes the drafts, it doesn't validate the checks."""
    rows = {r["question_id"]: r for r in read_answers(run_paths(writer, out_dir)[0])}
    retriever, questions = Retriever(split), split.sample(n, seed)
    counts: dict[str, int] = {}
    flagged_f1, clean_f1 = [], []
    for q in questions:
        row = rows[q.id]
        ids = {p.id for p in retriever.retrieve(q, writer.k)}
        found = checks.run(q.text, row["predicted_answer"], tuple(row["predicted_evidence"]), ids)
        for check in {f.check for f in found}:
            counts[check] = counts.get(check, 0) + 1
        if not is_unanswerable(row["predicted_answer"]) and normalize(row["predicted_answer"]) not in ("yes", "no"):
            (flagged_f1 if found else clean_f1).append(answer_f1(q, row["predicted_answer"])[0])
    return {"drafts": len(questions), "counts": dict(sorted(counts.items())), "flagged_f1": flagged_f1, "clean_f1": clean_f1}


def total_spent(out_dir: Path = RUNS_DIR) -> float:
    return sum(row["cost"] for path in out_dir.glob("*.jsonl") for row in read_answers(path))


def main() -> None:
    import anthropic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "run", "report"])
    parser.add_argument("runs", nargs="*", help="run names from configs/answering.toml; default: all")
    parser.add_argument("--dev", type=Path, default=DEV_SPLIT)
    parser.add_argument("--questions", type=int, help="size of the seeded question sample; default: the config's")
    parser.add_argument("--max-dollars", type=float, help="required for run: refuse if the estimate is higher")
    args = parser.parse_args()
    config = load_config("answering")["sample"]
    n = args.questions or config["questions"]
    split, costs, runs = QasperSplit.load(args.dev), load_costs(), Writer.load_all()
    client = anthropic.Anthropic() if args.command != "report" else None
    for name in args.runs or runs:
        run = runs[name]
        requests = build_requests(run, split, n, config["seed"])
        if args.command == "report":
            r = report(run, requests, split)
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
