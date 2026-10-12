"""The `rqa` command: one entry point for every step of the project. Only this module parses arguments and prints.

    uv run rqa download                 QASPER train/dev + the official evaluator, checksum-verified
    uv run rqa stats                    dev split summary
    uv run rqa recall                   BM25 evidence recall@k vs. a random baseline
"""

import argparse
from pathlib import Path

from reflective_qa.data.download import download
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.stats import summarize
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download and verify QASPER").set_defaults(func=_download)
    for name, func, help in [("stats", _stats, "summarize a split"), ("recall", _recall, "BM25 evidence recall@k")]:
        command = commands.add_parser(name, help=help)
        command.add_argument("--dev", type=Path, default=DEV_SPLIT, help="QASPER split file; default: dev")
        command.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)
