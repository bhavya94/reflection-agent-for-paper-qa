"""Score a set of predictions on a split's text-only questions, like the official evaluator: a missing prediction
scores 0 on both metrics, and only the questions asked about count."""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.metrics import answer_f1, evidence_f1, is_unanswerable


@dataclass(frozen=True)
class Prediction:
    answer: str
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict) -> "Prediction":
        """A row in the official evaluator's shape, {"predicted_answer", "predicted_evidence"}, but with paragraph
        IDs as evidence instead of paragraph text."""
        return cls(row["predicted_answer"], tuple(row["predicted_evidence"]))


def load_predictions(path: Path) -> dict[str, Prediction]:
    """JSON lines of {"question_id", "predicted_answer", "predicted_evidence"}."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["question_id"]: Prediction.from_row(r) for r in rows}


class Scorer:
    def __init__(self, split: QasperSplit):
        self.split = split

    def score(self, predictions: dict[str, Prediction], question_ids: set[str] | None = None) -> dict:
        """Scores every text-only question, or only those in `question_ids`, so questions outside a sample don't
        count as zeros."""
        questions = [q for q in self.split.text_only_questions() if question_ids is None or q.id in question_ids]
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
