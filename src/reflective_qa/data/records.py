"""Typed records for one QASPER split: papers, their paragraphs, questions and the annotators' answers."""

from dataclasses import dataclass
from enum import StrEnum

UNANSWERABLE = "Unanswerable"


class AnswerType(StrEnum):
    """Values match the official evaluator's type names."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    NONE = "none"


@dataclass(frozen=True)
class Paragraph:
    id: str
    section: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    type: AnswerType
    evidence_ids: tuple[str, ...]
    """Ids of the paragraphs the annotator selected as evidence."""
    other_evidence: tuple[str, ...]
    """Evidence that matches no paragraph. In QASPER v0.3 these are all section headings."""
    uses_figure_or_table: bool


@dataclass(frozen=True)
class Question:
    id: str
    paper_id: str
    text: str
    answers: tuple[Answer, ...]

    @property
    def text_only(self) -> bool:
        """True when no annotator needed a figure or table to answer."""
        return not any(answer.uses_figure_or_table for answer in self.answers)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    paragraphs: tuple[Paragraph, ...]
    questions: tuple[Question, ...]
