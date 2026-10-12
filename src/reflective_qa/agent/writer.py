"""The writer: the answer model's prompt, its revision prompt, and parsing its reply into an answer with cited
paragraph IDs."""

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any

from reflective_qa.agent.checks import Finding
from reflective_qa.data.records import Paper, Paragraph, Question
from reflective_qa.paths import load_config
from reflective_qa.retrieval.bm25 import passage_text

SYSTEM = """You answer questions about one scientific paper.
- Keep the answer as short as possible. When a phrase in the passages answers the question, copy it.
- For a yes/no question, answer "Yes" or "No".
- When passages are given, use only them. Each passage starts with its ID in brackets, for example [s3p0].
  If they do not contain the answer, answer exactly "Unanswerable".
- When no passages are given, give your best guess from what you know about the paper.
- In "evidence", list the IDs of the passages that support the answer, without brackets: s3p0, not [s3p0].
  List none if no passages are given or the answer is "Unanswerable"."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "evidence"],
    "additionalProperties": False,
}

# The answer rules are repeated because the 20-question pilot showed revisions drifting long and hedged
# ("Yes, partly: ..."), which the writer's system prompt alone did not prevent.
REVISE_INSTRUCTION = ("A reviewer found these problems with your draft. Fix them and answer again in the same format. "
                      "Keep the same rules: the shortest phrase that answers the question, Yes or No for a yes/no "
                      "question, and Unanswerable only if the passages do not contain the answer. If a problem is "
                      "wrong, keep that part of the draft.")


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    k: int
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class ModelAnswer:
    question_id: str
    model: str
    """The model that produced the reply, as the API reports it."""
    answer: str
    evidence_ids: tuple[str, ...]
    status: str
    """"ok", or why the answer is empty: "refusal", "max_tokens" or "invalid_json"."""
    input_tokens: int
    output_tokens: int

    def to_row(self, cost: float) -> dict:
        """The run file row, in the shape `load_predictions` reads."""
        return {"question_id": self.question_id, "predicted_answer": self.answer,
                "predicted_evidence": list(self.evidence_ids), "status": self.status, "model": self.model,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost": cost}


def passages_block(passages: list[Paragraph]) -> str:
    """Paragraphs as the writer and the critic see them: each labeled with its ID."""
    return "Passages:\n" + "\n\n".join(f"[{p.id}] {passage_text(p)}" for p in passages)


class Writer:
    """One writer run from configs/answering.toml: a model, how many paragraphs it sees, and its settings."""

    def __init__(self, config: RunConfig):
        self.config = config

    @classmethod
    def load_all(cls) -> dict[str, "Writer"]:
        config = load_config("answering")
        return {name: cls(RunConfig(name=name, max_tokens=config["max_tokens"], **run)) for name, run in config["runs"].items()}

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def k(self) -> int:
        return self.config.k

    @cached_property
    def fingerprint(self) -> str:
        """A hash of the run config, prompt and schema. Answers are filed under it, so changing any of them starts a
        new file instead of mixing answers from two setups."""
        setup = {"run": {k: v for k, v in asdict(self.config).items() if k != "name"}, "system": SYSTEM,
                 "schema": ANSWER_SCHEMA}
        return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:10]

    def request(self, paper: Paper, question: Question, passages: list[Paragraph]) -> dict[str, Any]:
        """Messages API parameters; the same dict works for a direct call and for a Batch API request."""
        if passages:
            context = passages_block(passages)
        else:
            context = "No passages are given. Give your best guess from what you know about this paper."
        return {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": f"Paper: {paper.title}\n\n{context}\n\nQuestion: {question.text}"}],
            "output_config": {"effort": self.config.effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        }

    def revise_request(self, paper: Paper, question: Question, passages: list[Paragraph], draft: dict,
                       found: list[Finding]) -> dict[str, Any]:
        """The original request, plus the draft and the problems to fix, in one user turn."""
        request = self.request(paper, question, passages)
        answer = json.dumps({"answer": draft["predicted_answer"], "evidence": draft["predicted_evidence"]})
        feedback = "\n".join(f"- {f.check}: {f.detail}" for f in found)
        content = f"{request['messages'][0]['content']}\n\nYour draft:\n{answer}\n\n{REVISE_INSTRUCTION}\n{feedback}"
        return request | {"messages": [{"role": "user", "content": content}]}

    @staticmethod
    def parse(question_id: str, message: Any) -> ModelAnswer:
        """Cited IDs lose any brackets or spaces around them; an ID that wasn't retrieved is kept, and scoring counts
        it as a wrong citation."""
        usage = (message.usage.input_tokens, message.usage.output_tokens)
        if message.stop_reason in ("refusal", "max_tokens"):
            return ModelAnswer(question_id, message.model, "", (), message.stop_reason, *usage)
        text = next((block.text for block in message.content if block.type == "text"), "")
        try:
            reply = json.loads(text)
            evidence = tuple(str(cited).strip().strip("[]").strip() for cited in reply["evidence"])
            return ModelAnswer(question_id, message.model, reply["answer"], evidence, "ok", *usage)
        except (json.JSONDecodeError, KeyError, TypeError):
            return ModelAnswer(question_id, message.model, "", (), "invalid_json", *usage)
