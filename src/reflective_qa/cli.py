"""The `rqa` command: one entry point for every step of the project. Only this module parses arguments and prints.

    uv run rqa download                 QASPER train/dev + the official evaluator, checksum-verified
    uv run rqa stats                    dev split summary
    uv run rqa recall                   BM25 evidence recall@k vs. a random baseline
    uv run rqa score PREDICTIONS        answer-F1 (official) and paragraph-ID evidence-F1 of a JSON lines file
    uv run rqa baselines report         one-shot answers of every writer run (estimate and run need an API key)
    uv run rqa checks                   how often each code check fires on the first answers
    uv run rqa reflect report           every reflection setup against its drafts (estimate and run need an API key)

Paid commands need ANTHROPIC_API_KEY (`uv run --env-file .env rqa ...`) and refuse to run without --max-dollars.
"""

import argparse
import statistics
from pathlib import Path
from typing import Any

from reflective_qa.agent.checks import CodeChecks
from reflective_qa.agent.reflection import ReflectionAgent
from reflective_qa.agent.writer import Writer
from reflective_qa.data.download import download
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.stats import summarize
from reflective_qa.eval.scorer import Scorer, load_predictions
from reflective_qa.experiments.baseline import BaselineExperiment
from reflective_qa.experiments.reflection import ReflectionExperiment
from reflective_qa.paths import DEV_SPLIT, load_config
from reflective_qa.retrieval.recall import RecallEvaluator
from reflective_qa.runs.batch import OverBudget
from reflective_qa.runs.store import RunStore


def _download(args: argparse.Namespace) -> None:
    print(f"QASPER files in {download()}")


def _stats(args: argparse.Namespace) -> None:
    for key, value in summarize(QasperSplit.load(args.dev)).items():
        print(f"{key:32} {value}")


def _recall(args: argparse.Namespace) -> None:
    config = load_config("retrieval")
    rows = RecallEvaluator(QasperSplit.load(args.dev)).evaluate(config["report_ks"], config["random_samples"], config["seed"])
    print(f"{rows[0]['questions']} text-only questions with paragraph evidence")
    print(f"{'k':>3}  {'recall':>6}  {'all found':>9}  {'random recall':>13}  {'random all found':>16}"
          f"  {'context chars (median)':>22}")
    for row in rows:
        print(f"{row['k']:>3}  {row['recall']:>6.2f}  {row['all_found']:>9.2f}  {row['random_recall']:>13.2f}"
              f"  {row['random_all_found']:>16.2f}  {row['median_chars']:>22,.0f}")


def _score(args: argparse.Namespace) -> None:
    result = Scorer(QasperSplit.load(args.dev)).score(load_predictions(args.predictions))
    print(f"{result['questions']} text-only questions, {result['missing']} without a prediction")
    print(f"answer-F1 {result['answer_f1']:.3f}  predicted 'Unanswerable' {result['predicted_unanswerable']:.1%}")
    print(f"paragraph-ID evidence-F1 {result['evidence_f1']:.3f} on {result['evidence_questions']} questions with "
          f"cited paragraphs (not comparable to published QASPER evidence-F1)")
    for answer_type, f1 in result["answer_f1_by_type"].items():
        print(f"  {answer_type:12} {f1:.3f}")


def _client(args: argparse.Namespace) -> Any | None:
    """An API client only for the commands that call the API; `report` reads files only."""
    if args.action == "report":
        return None
    if args.action == "run" and args.max_dollars is None:
        raise SystemExit("run needs --max-dollars")
    import anthropic
    return anthropic.Anthropic()


def _baselines(args: argparse.Namespace) -> None:
    store, writers = RunStore(), Writer.load_all()
    experiment = BaselineExperiment.from_config(QasperSplit.load(args.dev), store, args.questions, _client(args))
    for name in args.names or writers:
        writer = writers[name]
        requests = experiment.requests(writer)
        if args.action == "report":
            r = experiment.report(writer, requests)
            by_type = "  ".join(f"{t} {f1:.3f}" for t, f1 in r["answer_f1_by_type"].items())
            print(f"{r['run']:20} {r['questions'] - r['missing']}/{r['questions']} answered  answer-F1 {r['answer_f1']:.3f}  "
                  f"evidence-F1 {r['evidence_f1']:.3f}  unanswerable {r['predicted_unanswerable']:.1%}  "
                  f"{r['output_tokens']:.0f} output tokens  ${r['cost']:.2f}\n{'':22}{by_type}")
            continue
        e = experiment.estimate(writer, experiment.pending(writer, requests))
        print(f"{writer.name:20} {len(requests)} questions, {e['questions']} unanswered  ~{e['input_tokens']:,.0f} input "
              f"tokens  {e['output_tokens']:,.0f} output ({'measured' if e['output_measured'] else 'assumed'})  "
              f"≈ ${e['dollars']:.2f}")
        if args.action == "run":
            path = experiment.run(writer, requests, args.max_dollars)
            rows = store.read(path)
            statuses = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
            print(f"  -> {path.name}: {len(rows)} answers {statuses}, "
                  f"${sum(r['cost'] for r in rows):.2f} for this run, ${store.total_spent():.2f} across all runs")


def _checks(args: argparse.Namespace) -> None:
    experiment, checks = BaselineExperiment.from_config(QasperSplit.load(args.dev), RunStore()), CodeChecks.from_config()
    mean = lambda xs: f"{statistics.mean(xs):.3f} (n={len(xs)})" if xs else "n/a"
    for name, writer in Writer.load_all().items():
        if not writer.k:
            continue  # closed-book drafts cite nothing, so there is nothing to check against
        r = experiment.check_findings(writer, checks)
        print(f"{name}: {sum(r['counts'].values())} findings on {r['drafts']} drafts  {r['counts']}")
        print(f"  content answers: answer-F1 flagged {mean(r['flagged_f1'])} vs not flagged {mean(r['clean_f1'])}")


def _reflect(args: argparse.Namespace) -> None:
    experiment = ReflectionExperiment.from_config(QasperSplit.load(args.dev), RunStore(), args.questions, _client(args))
    for name, agent in ReflectionAgent.load_all().items():
        if args.names and name not in args.names:
            continue
        contexts = experiment.contexts(agent)
        critiques, revisions = experiment.results(agent)
        if args.action == "estimate":
            e = experiment.estimate(agent, contexts, critiques, revisions)
            print(f"{name:28} critic {e['critic_requests']} requests ≈ ${e['critic_dollars']:.2f}   "
                  f"revise {e['revise_requests']} requests ≈ ${e['revise_dollars']:.2f}")
        elif args.action == "run":
            experiment.run(agent, contexts, args.max_dollars)
            print(f"{name}: done; run `report` to score it")
        else:
            r = experiment.report(contexts, critiques, revisions)
            a = experiment.analyze(contexts, critiques, revisions)
            ran = experiment.has_run(agent, contexts, critiques, revisions)
            status = (f"critic: {r['critiques']} reviews ({r['critic_revise']} revise, {r['critic_errors']} errors)  "
                      f"revisions: {r['revisions']} ({r['revision_failures']} failed), {r['changed']} changed: answer-F1 "
                      f"better {r['better']}, worse {r['worse']}") if ran else "not run yet"
            print(f"{name:28} answer-F1 {r['draft_answer_f1']:.3f} -> {r['answer_f1']:.3f}  evidence-F1 "
                  f"{r['evidence_f1']:.3f}  unanswerable {r['predicted_unanswerable']:.1%}  ${r['cost']:.2f}\n{'':30}{status}")
            if ran:
                kinds = "  ".join(f"{k.replace('_', ' ')} {v['n']} ({v['mean_delta']:+.3f})" for k, v in a["kinds"].items())
                print(f"{'':30}change {a['delta']:+.3f}, 95% CI [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]  {kinds}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download and verify QASPER").set_defaults(func=_download)
    handlers = {"stats": (_stats, "summarize a split"), "recall": (_recall, "BM25 evidence recall@k"),
                "score": (_score, "score a predictions file"), "checks": (_checks, "code checks on the first answers"),
                "baselines": (_baselines, "one-shot answers: estimate, run or report"),
                "reflect": (_reflect, "one round of reflection: estimate, run or report")}
    for name, (func, help) in handlers.items():
        command = commands.add_parser(name, help=help)
        command.add_argument("--dev", type=Path, default=DEV_SPLIT, help="QASPER split file; default: dev")
        command.set_defaults(func=func)
    commands.choices["score"].add_argument("predictions", type=Path,
                                           help="JSON lines of question_id, predicted_answer, predicted_evidence")
    for name, configured, budget in [
            ("baselines", "writer runs in configs/answering.toml", "per writer run"),
            ("reflect", "setups in configs/reflection.toml", "per setup and per stage, so 6 setups x 2 stages can spend 12x")]:
        command = commands.choices[name]
        command.add_argument("action", choices=["estimate", "run", "report"])
        command.add_argument("names", nargs="*", help=f"{configured}; default: all")
        command.add_argument("--questions", type=int, help="size of the seeded question sample; default: the config's")
        command.add_argument("--max-dollars", type=float, help=f"required for run: refuse if an estimate is higher; {budget}")
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except OverBudget as refused:
        raise SystemExit(str(refused)) from None
