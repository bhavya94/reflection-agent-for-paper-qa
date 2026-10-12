"""Answer-F1 and evidence-F1 for predictions on QASPER's text-only questions.

Answer-F1 follows the official evaluator (SQuAD normalization, token F1, best reference wins), scored only on
the text-only questions, so questions outside the working set don't count as zeros.

Evidence-F1 here is our own metric and not comparable to published QASPER numbers: it compares paragraph IDs,
not raw evidence strings, and it scores only questions where some annotator cited a paragraph. Questions whose
references cite nothing (unanswerable, or heading-only evidence) are left out, so citing nothing can't score.
"""

import json
import re
import statistics
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.data.records import UNANSWERABLE, Question


@dataclass(frozen=True)
class Prediction:
    answer: str
    evidence_ids: tuple[str, ...]


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


def score(split: QasperSplit, predictions: dict[str, Prediction], question_ids: set[str] | None = None) -> dict:
    """Scores every text-only question, or only those in `question_ids`; a missing prediction scores 0 on both
    metrics, like the official evaluator."""
    questions = [q for q in split.text_only_questions() if question_ids is None or q.id in question_ids]
    answer_scores, evidence_scores, by_type = [], [], {}
    for question in questions:
        prediction = predictions.get(question.id)
        f1, answer_type = answer_f1(question, prediction.answer) if prediction else (0.0, "missing")
        answer_scores.append(f1)
        if (evidence := evidence_f1(question, prediction.evidence_ids if prediction else ())) is not None:
            evidence_scores.append(evidence)
        by_type.setdefault(answer_type, []).append(f1)
    scored = [predictions[q.id] for q in questions if q.id in predictions]
    return {
        "questions": len(answer_scores),
        "missing": len(answer_scores) - len(scored),
        "answer_f1": statistics.mean(answer_scores),
        "answer_f1_by_type": {t: statistics.mean(v) for t, v in sorted(by_type.items()) if t != "missing"},
        "evidence_f1": statistics.mean(evidence_scores),
        "evidence_questions": len(evidence_scores),
        "predicted_unanswerable": statistics.mean(is_unanswerable(p.answer) for p in scored) if scored else 0.0,
    }


def load_predictions(path: Path) -> dict[str, Prediction]:
    """JSON lines of {"question_id", "predicted_answer", "predicted_evidence"}: the official evaluator's shape, but
    evidence is paragraph IDs, not paragraph text."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["question_id"]: Prediction(r["predicted_answer"], tuple(r["predicted_evidence"])) for r in rows}


if __name__ == "__main__":
    result = score(QasperSplit.load(Path(sys.argv[1])), load_predictions(Path(sys.argv[2])))
    print(f"{result['questions']} text-only questions, {result['missing']} without a prediction")
    print(f"answer-F1 {result['answer_f1']:.3f}  predicted 'Unanswerable' {result['predicted_unanswerable']:.1%}")
    print(f"paragraph-ID evidence-F1 {result['evidence_f1']:.3f} on {result['evidence_questions']} questions with "
          f"cited paragraphs (not comparable to published QASPER evidence-F1)")
    for answer_type, f1 in result["answer_f1_by_type"].items():
        print(f"  {answer_type:12} {f1:.3f}")
