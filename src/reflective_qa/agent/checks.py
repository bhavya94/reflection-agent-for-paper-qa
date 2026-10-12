"""Code checks on a draft answer: free, deterministic feedback (Module 2's "external feedback") for the critic and
the writer. Each finding uses a name from the critic's checklist, so the critic can confirm or reject it."""

import re
from dataclasses import dataclass

from reflective_qa.eval.metrics import is_unanswerable, normalize
from reflective_qa.paths import load_config

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


class CodeChecks:
    def __init__(self, max_words: int):
        self.max_words = max_words

    @classmethod
    def from_config(cls) -> "CodeChecks":
        return cls(load_config("reflection")["max_answer_words"])

    def run(self, question: str, answer: str, cited: tuple[str, ...], passage_ids: set[str]) -> list[Finding]:
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
        if not abstained and (words := len(answer.split())) > self.max_words:
            findings.append(Finding("too_long", f"{words} words. Use the shortest phrase that answers the question."))
        return findings
