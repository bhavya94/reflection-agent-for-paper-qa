"""Answer-F1 and evidence-F1 for predictions on QASPER's text-only questions.

Answer-F1 follows the official evaluator (SQuAD normalization, token F1, best reference wins), scored only on
the text-only questions, so questions outside the working set don't count as zeros. Evidence-F1 compares
paragraph IDs rather than raw evidence strings, so whitespace differences and section headings in the gold
evidence don't count as misses.
"""

import json
import re
import statistics
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.data import UNANSWERABLE, Paper, Question, text_only_questions


@dataclass(frozen=True)
class Prediction:
    answer: str
    evidence_ids: tuple[str, ...]


def normalize(text: str) -> str:
    """SQuAD v1.1 normalization, in the official order: lowercase, punctuation, articles, whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
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
    best = max(question.answers, key=lambda reference: token_f1(answer, reference.text))
    return token_f1(answer, best.text), best.type.value


def evidence_f1(question: Question, evidence_ids: tuple[str, ...]) -> float:
    """Best paragraph-set F1 over the references; citing nothing for a reference with no evidence scores 1."""
    scores = []
    for reference in question.answers:
        gold, predicted = set(reference.evidence_ids), set(evidence_ids)
        if not gold and not predicted:
            scores.append(1.0)
        elif not gold & predicted:
            scores.append(0.0)
        else:
            precision, recall = len(gold & predicted) / len(predicted), len(gold & predicted) / len(gold)
            scores.append(2 * precision * recall / (precision + recall))
    return max(scores)


def score(papers: list[Paper], predictions: dict[str, Prediction]) -> dict:
    """Scores every text-only question; a missing prediction scores 0 on both metrics, like the official evaluator."""
    questions = text_only_questions(papers)
    answer_scores, evidence_scores, by_type = [], [], {}
    for question in questions:
        prediction = predictions.get(question.id)
        f1, answer_type = answer_f1(question, prediction.answer) if prediction else (0.0, "missing")
        answer_scores.append(f1)
        evidence_scores.append(evidence_f1(question, prediction.evidence_ids) if prediction else 0.0)
        by_type.setdefault(answer_type, []).append(f1)
    scored = [predictions[q.id] for q in questions if q.id in predictions]
    return {
        "questions": len(answer_scores),
        "missing": len(answer_scores) - len(scored),
        "answer_f1": statistics.mean(answer_scores),
        "answer_f1_by_type": {t: statistics.mean(v) for t, v in sorted(by_type.items()) if t != "missing"},
        "evidence_f1": statistics.mean(evidence_scores),
        "predicted_unanswerable": statistics.mean(p.answer.strip() == UNANSWERABLE for p in scored) if scored else 0.0,
    }


def load_predictions(path: Path) -> dict[str, Prediction]:
    """JSON lines of {"question_id", "predicted_answer", "predicted_evidence": [paragraph IDs]}."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["question_id"]: Prediction(r["predicted_answer"], tuple(r["predicted_evidence"])) for r in rows}


if __name__ == "__main__":
    from reflective_qa.data import load_split

    result = score(load_split(Path(sys.argv[1])), load_predictions(Path(sys.argv[2])))
    print(f"{result['questions']} text-only questions, {result['missing']} without a prediction")
    print(f"answer-F1 {result['answer_f1']:.3f}  evidence-F1 {result['evidence_f1']:.3f}"
          f"  predicted 'Unanswerable' {result['predicted_unanswerable']:.1%}")
    for answer_type, f1 in result["answer_f1_by_type"].items():
        print(f"  {answer_type:12} {f1:.3f}")
