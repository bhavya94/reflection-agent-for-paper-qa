"""The critic: reviews a draft answer against the question and the paragraphs the writer saw, then passes it or
lists problems from a fixed checklist that the writer can act on."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from reflective_qa.agent.checks import Finding
from reflective_qa.agent.writer import passages_block
from reflective_qa.data.records import Paper, Paragraph, Question

CHECKLIST = {
    "unsupported": "A claim in the answer is not supported by the cited passages.",
    "wrong_or_incomplete": "The answer is wrong, or misses part of what the question asks.",
    "wrongly_unanswerable": "The answer says Unanswerable, but the passages contain the answer.",
    "should_be_unanswerable": "The answer states something, but the passages do not contain the answer.",
    "too_long": "The answer is longer than the question needs; the shortest correct phrase is better.",
    "bad_citation": "The cited passages are not the ones that support the answer.",
    "yes_no_format": "A yes/no question is not answered with Yes or No.",
}

SYSTEM = "\n".join([
    "You review a draft answer to a question about one scientific paper.",
    "- Judge only from the passages in the message. They are everything the writer saw.",
    "- Check the draft against each item below:",
    *(f"  - {name}: {meaning}" for name, meaning in CHECKLIST.items()),
    '- If no item applies, return verdict "pass" and no problems.',
    '- Otherwise return verdict "revise" and one problem per item that applies. In "detail", say what to change so',
    "  the writer can fix it; when a passage shows it, name that passage's ID.",
    "- Automatic check results, when given, can be wrong. Confirm or reject each one; report only real problems.",
])

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "problems": {"type": "array", "items": {
            "type": "object",
            "properties": {"check": {"type": "string", "enum": list(CHECKLIST)}, "detail": {"type": "string"}},
            "required": ["check", "detail"],
            "additionalProperties": False,
        }},
    },
    "required": ["verdict", "problems"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CriticConfig:
    model: str
    effort: str
    max_tokens: int
    code_feedback: bool
    """Whether the critic sees the code checks' findings."""


@dataclass(frozen=True)
class Critique:
    question_id: str
    model: str
    verdict: str
    """"revise" only when the critic listed problems; "pass" otherwise; "error" if the reply was unusable."""
    problems: tuple[Finding, ...]
    status: str
    input_tokens: int
    output_tokens: int

    def to_row(self, cost: float) -> dict:
        return {"question_id": self.question_id, "verdict": self.verdict, "problems": [asdict(f) for f in self.problems],
                "status": self.status, "model": self.model, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost": cost}


class Critic:
    def __init__(self, config: CriticConfig):
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    def request(self, paper: Paper, question: Question, passages: list[Paragraph], answer: str, cited: tuple[str, ...],
                findings: list[Finding]) -> dict[str, Any]:
        """Messages API parameters for one review; works for a direct call and for a Batch API request."""
        content = (f"Paper: {paper.title}\n\n{passages_block(passages)}\n\nQuestion: {question.text}\n\n"
                   f"Draft answer: {answer}\nCited passages: {', '.join(cited) or 'none'}")
        if self.config.code_feedback:
            lines = [f"- {f.check}: {f.detail}" for f in findings] or ["- none flagged"]
            content += "\n\nAutomatic checks:\n" + "\n".join(lines)
        return {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.config.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        }

    @staticmethod
    def parse(question_id: str, message: Any) -> Critique:
        """A "revise" verdict without problems gives the writer nothing to act on, so it counts as "pass"."""
        usage = (message.usage.input_tokens, message.usage.output_tokens)
        if message.stop_reason in ("refusal", "max_tokens"):
            return Critique(question_id, message.model, "error", (), message.stop_reason, *usage)
        text = next((block.text for block in message.content if block.type == "text"), "")
        try:
            reply = json.loads(text)
            problems = tuple(Finding(p["check"], p["detail"]) for p in reply["problems"])
            verdict = "revise" if reply["verdict"] == "revise" and problems else "pass"
            return Critique(question_id, message.model, verdict, problems if verdict == "revise" else (), "ok", *usage)
        except (json.JSONDecodeError, KeyError, TypeError):
            return Critique(question_id, message.model, "error", (), "invalid_json", *usage)
