"""The answer model's prompt, and parsing of its reply into an answer with cited paragraph IDs."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflective_qa.data import Paper, Paragraph, Question
from reflective_qa.retrieval import passage_text

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "answering.toml"

SYSTEM = """You answer questions about one scientific paper.
- Keep the answer as short as possible. When a phrase in the passages answers the question, copy it.
- For a yes/no question, answer "Yes" or "No".
- When passages are given, use only them. Each passage starts with its ID in brackets, for example [s3p0].
  If they do not contain the answer, answer exactly "Unanswerable".
- When no passages are given, give your best guess from what you know about the paper.
- In "evidence", list the IDs of the passages that support the answer. List none if no passages are given or the
  answer is "Unanswerable"."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "evidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    k: int
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class Answer:
    question_id: str
    model: str
    """The model that produced the reply, as the API reports it."""
    answer: str
    evidence_ids: tuple[str, ...]
    status: str
    """"ok", or why the answer is empty: "refusal", "max_tokens" or "invalid_json"."""
    input_tokens: int
    output_tokens: int


def load_runs(path: Path = CONFIG) -> dict[str, RunConfig]:
    config = tomllib.loads(path.read_text())
    return {name: RunConfig(name=name, max_tokens=config["max_tokens"], **run) for name, run in config["runs"].items()}


def build_request(run: RunConfig, paper: Paper, question: Question, passages: list[Paragraph]) -> dict[str, Any]:
    """Messages API parameters; the same dict works for a direct call and for a Batch API request."""
    if passages:
        context = "Passages:\n" + "\n\n".join(f"[{p.id}] {passage_text(p)}" for p in passages)
    else:
        context = "No passages are given. Give your best guess from what you know about this paper."
    return {
        "model": run.model,
        "max_tokens": run.max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Paper: {paper.title}\n\n{context}\n\nQuestion: {question.text}"}],
        "output_config": {"effort": run.effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
    }


def parse_message(question_id: str, message: Any) -> Answer:
    """Citations are kept as the model wrote them; scoring treats an ID that wasn't retrieved as a wrong citation."""
    usage = (message.usage.input_tokens, message.usage.output_tokens)
    if message.stop_reason in ("refusal", "max_tokens"):
        return Answer(question_id, message.model, "", (), message.stop_reason, *usage)
    text = next((block.text for block in message.content if block.type == "text"), "")
    try:
        reply = json.loads(text)
        return Answer(question_id, message.model, reply["answer"], tuple(reply["evidence"]), "ok", *usage)
    except (json.JSONDecodeError, KeyError, TypeError):
        return Answer(question_id, message.model, "", (), "invalid_json", *usage)
