"""Code checks on a draft answer: free, deterministic feedback (Module 2's "external feedback") for the critic and
the writer. Each finding uses a name from the critic's checklist, so the critic can confirm or reject it."""

import re
import statistics
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.scoring import is_unanswerable, normalize

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "reflection.toml"
# Starts with an auxiliary verb and offers no alternatives ("X or Y?"). On train: 97% recall, 0.82 precision for
# questions all annotators answered Yes or No; most other matches were marked unanswerable.
_YES_NO_QUESTION = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", re.I)


@dataclass(frozen=True)
class Finding:
    check: str
    detail: str


def is_yes_no_question(question: str) -> bool:
    question = question.strip()
    return bool(_YES_NO_QUESTION.match(question)) and " or " not in question.lower()


def run_checks(question: str, answer: str, cited: tuple[str, ...], passage_ids: set[str],
               max_words: int) -> list[Finding]:
    """`passage_ids` are the paragraphs the writer was given. With none (closed-book), citations aren't checked.
    An empty answer (a refusal or an unusable reply) gets no findings: there is nothing to give feedback on."""
    if not answer.strip():
        return []
    findings = []
    abstained = is_unanswerable(answer)
    if passage_ids:
        if unknown := sorted({pid for pid in cited if pid not in passage_ids}):
            verb = "was" if len(unknown) == 1 else "were"
            findings.append(Finding("bad_citation", f"Cites {', '.join(unknown)}, which {verb} not among the given paragraphs."))
        if not abstained and not cited:
            findings.append(Finding("bad_citation", "Gives an answer but cites no paragraph."))
        if abstained and cited:
            findings.append(Finding("bad_citation", "Says Unanswerable but cites paragraphs. An Unanswerable answer cites none."))
    if is_yes_no_question(question) and not abstained and normalize(answer) not in ("yes", "no"):
        findings.append(Finding("yes_no_format", "This looks like a yes/no question. If it is, answer Yes or No."))
    if not abstained and (words := len(answer.split())) > max_words:
        findings.append(Finding("too_long", f"{words} words. Use the shortest phrase that answers the question."))
    return findings


def load_max_words(path: Path = CONFIG) -> int:
    return tomllib.loads(path.read_text())["max_answer_words"]


if __name__ == "__main__":
    # How often each check fires on the M1 drafts. The F1 comparison covers content answers only (not Unanswerable,
    # Yes or No), since too_long can't flag those; it describes the drafts, it doesn't validate the checks.
    from reflective_qa.answering import CONFIG as ANSWERING, load_runs
    from reflective_qa.baselines import RUNS_DIR, read_answers, run_paths, sample_questions
    from reflective_qa.data import load_split, text_only_questions
    from reflective_qa.retrieval import PaperRetriever
    from reflective_qa.scoring import answer_f1

    papers = load_split(Path(sys.argv[1]))
    by_id, max_words = {p.id: p for p in papers}, load_max_words()
    sample = tomllib.loads(ANSWERING.read_text())["sample"]
    questions = sample_questions(text_only_questions(papers), sample["questions"], sample["seed"])
    mean = lambda xs: f"{statistics.mean(xs):.3f} (n={len(xs)})" if xs else "n/a"
    for name, run in load_runs().items():
        if not run.k:
            continue  # closed-book drafts cite nothing, so there is nothing to check against
        rows = {r["question_id"]: r for r in read_answers(run_paths(run, RUNS_DIR)[0])}
        counts, flagged_f1, clean_f1 = {}, [], []
        for q in questions:
            row = rows[q.id]
            ids = {p.id for p in PaperRetriever(by_id[q.paper_id]).retrieve(q.text, run.k)}
            found = run_checks(q.text, row["predicted_answer"], tuple(row["predicted_evidence"]), ids, max_words)
            for check in {f.check for f in found}:
                counts[check] = counts.get(check, 0) + 1
            if not is_unanswerable(row["predicted_answer"]) and normalize(row["predicted_answer"]) not in ("yes", "no"):
                (flagged_f1 if found else clean_f1).append(answer_f1(q, row["predicted_answer"])[0])
        print(f"{name}: {sum(counts.values())} findings on {len(questions)} drafts  {dict(sorted(counts.items()))}")
        print(f"  content answers: answer-F1 flagged {mean(flagged_f1)} vs not flagged {mean(clean_f1)}")
