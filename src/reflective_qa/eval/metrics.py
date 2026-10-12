"""Answer-F1 and evidence-F1 for one answer to one question. Pure functions: no state, no files.

Answer-F1 follows the official evaluator: SQuAD normalization, token F1, best reference wins.

Evidence-F1 here is our own metric and not comparable to published QASPER numbers: it compares paragraph IDs,
not raw evidence strings, and it scores only questions where some annotator cited a paragraph. Questions whose
references cite nothing (unanswerable, or heading-only evidence) are left out, so citing nothing can't score.
"""

import re
import string
from collections import Counter

from reflective_qa.data.records import UNANSWERABLE, Question

_NO_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD v1.1 normalization, in the official order: lowercase, punctuation, articles, whitespace."""
    text = text.lower().translate(_NO_PUNCTUATION)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    predicted, gold = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def answer_f1(question: Question, answer: str) -> tuple[float, str]:
    """Best F1 over the references, with the answer type of the reference that gave it (first one on ties)."""
    scores = [(token_f1(answer, reference.text), reference.type.value) for reference in question.answers]
    return max(scores, key=lambda pair: pair[0])


def evidence_f1(question: Question, evidence_ids: tuple[str, ...]) -> float | None:
    """Best paragraph-set F1 over the references that cite paragraphs; None if none do. Repeated IDs count once."""
    predicted, scores = set(evidence_ids), []
    for gold in (set(reference.evidence_ids) for reference in question.answers if reference.evidence_ids):
        overlap = len(gold & predicted)
        scores.append(0.0 if not overlap else 2 * overlap / (len(predicted) + len(gold)))
    return max(scores) if scores else None


def is_unanswerable(answer: str) -> bool:
    """Same normalization as answer-F1, so "unanswerable." counts as an abstention too."""
    return normalize(answer) == normalize(UNANSWERABLE)
