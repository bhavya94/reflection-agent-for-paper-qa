"""The `rqa` command: one entry point for every step of the project. Only this module parses arguments and prints.

    uv run rqa download                 QASPER train/dev + the official evaluator, checksum-verified
    uv run rqa stats                    dev split summary
    uv run rqa recall                   BM25 evidence recall@k vs. a random baseline
    uv run rqa score PREDICTIONS        answer-F1 (official) and paragraph-ID evidence-F1 of a JSON lines file
"""

import argparse
from pathlib import Path

from reflective_qa.data.download import download
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.stats import summarize
from reflective_qa.eval.scorer import Scorer, load_predictions
from reflective_qa.paths import DEV_SPLIT, load_config
from reflective_qa.retrieval.recall import RecallEvaluator


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download and verify QASPER").set_defaults(func=_download)
    split_commands = {name: commands.add_parser(name, help=help) for name, help in [
        ("stats", "summarize a split"), ("recall", "BM25 evidence recall@k"), ("score", "score a predictions file")]}
    split_commands["score"].add_argument("predictions", type=Path, help="JSON lines of question_id, predicted_answer, "
                                         "predicted_evidence")
    for name, command in split_commands.items():
        command.add_argument("--dev", type=Path, default=DEV_SPLIT, help="QASPER split file; default: dev")
        command.set_defaults(func={"stats": _stats, "recall": _recall, "score": _score}[name])
    args = parser.parse_args(argv)
    args.func(args)
