"""One round of reflection for one setup: code checks and an optional critic flag problems in a draft, the writer
revises the flagged drafts once, and the revision replaces the draft if it succeeded."""

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

from reflective_qa.agent.checks import CodeChecks, Finding
from reflective_qa.agent.critic import Critic, CriticConfig
from reflective_qa.agent.writer import Writer
from reflective_qa.data.records import Paper, Paragraph, Question
from reflective_qa.eval.analysis import Revision
from reflective_qa.paths import load_config


@dataclass(frozen=True)
class Setup:
    name: str
    writer: str
    """A run in configs/answering.toml: its answers are the drafts, and its model writes the revision."""
    critic: CriticConfig | None
    """None means code feedback only: the checks' findings go straight to the writer."""

    @classmethod
    def load_all(cls) -> dict[str, "Setup"]:
        config = load_config("reflection")
        setups = {}
        for name, s in config["setups"].items():
            critic = None if s["critic"] == "none" else CriticConfig(s["critic"], config["critic"]["effort"],
                                                                    config["critic"]["max_tokens"], s["code_feedback"])
            setups[name] = cls(name, s["writer"], critic)
        return setups


@dataclass(frozen=True)
class Context:
    """Everything one question's reflection needs: what the writer saw, what it answered, what the checks found."""
    paper: Paper
    question: Question
    passages: list[Paragraph]
    draft: dict
    findings: list[Finding]


class ReflectionAgent:
    def __init__(self, setup: Setup, writer: Writer, checks: CodeChecks):
        self.setup, self.writer, self.checks = setup, writer, checks
        self.critic = Critic(setup.critic) if setup.critic else None

    @classmethod
    def load_all(cls) -> dict[str, "ReflectionAgent"]:
        """One agent per setup in configs/reflection.toml, with its writer run and the configured code checks."""
        writers, checks = Writer.load_all(), CodeChecks.from_config()
        return {name: cls(setup, writers[setup.writer], checks) for name, setup in Setup.load_all().items()}

    @property
    def name(self) -> str:
        return self.setup.name

    def context(self, paper: Paper, question: Question, passages: list[Paragraph], draft: dict) -> Context:
        findings = self.checks.run(question.text, draft["predicted_answer"], tuple(draft["predicted_evidence"]),
                                   {p.id for p in passages})
        return Context(paper, question, passages, draft, findings)

    def critic_requests(self, contexts: dict[str, Context]) -> dict[str, dict]:
        if self.critic is None:
            return {}
        return {qid: self.critic.request(c.paper, c.question, c.passages, c.draft["predicted_answer"],
                                         tuple(c.draft["predicted_evidence"]), c.findings) for qid, c in contexts.items()}

    def problems(self, context: Context, critique: dict | None) -> list[Finding]:
        """What the writer is asked to fix: the critic's problems, or with no critic, the checks' findings."""
        if self.critic is None:
            return context.findings
        if critique is None or critique["verdict"] != "revise":
            return []
        return [Finding(p["check"], p["detail"]) for p in critique["problems"]]

    def revise_request(self, context: Context, found: list[Finding]) -> dict:
        return self.writer.revise_request(context.paper, context.question, context.passages, context.draft, found)

    def revise_requests(self, contexts: dict[str, Context], critiques: dict[str, dict]) -> dict[str, dict]:
        """Only drafts with something to fix are revised."""
        return {qid: self.revise_request(c, found) for qid, c in contexts.items()
                if (found := self.problems(c, critiques.get(qid)))}

    @cached_property
    def fingerprint(self) -> str:
        """A hash of the actual critic and revise requests for a fixed probe question, so any change to a prompt,
        schema, model setting, finding text or check behavior starts new files. The word limit counts only where
        findings are shown: a critic that never sees them keeps its paid reviews when the limit changes."""
        probe = self._probe()
        critique = {"verdict": "revise", "problems": [{"check": "unsupported", "detail": "probe"}]}
        parts = {"writer": self.writer.fingerprint, "critic_requests": self.critic_requests({"probe": probe}),
                 "revise_request": self.revise_request(probe, self.problems(probe, critique))}
        if self.setup.critic is None or self.setup.critic.code_feedback:
            parts["max_words"] = self.checks.max_words  # it decides which drafts get findings
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]

    def _probe(self) -> Context:
        """A fixed question whose draft trips several checks, so the fingerprint sees real finding texts."""
        paragraphs = [Paragraph("s0p0", "Probe", "A probe paragraph."), Paragraph("s1p0", "Probe", "Another probe paragraph.")]
        question = Question("probe", "probe", "Do they use a probe?", ())
        answer = " ".join(["word"] * 60)  # fixed, so the draft text never depends on the word limit
        draft = {"question_id": "probe", "predicted_answer": answer, "predicted_evidence": ["s9p9"], "status": "ok"}
        return self.context(Paper("probe", "Probe paper", "", tuple(paragraphs), (question,)), question, paragraphs, draft)

    @staticmethod
    def final_answers(contexts: dict[str, Context], critiques: dict[str, dict], revisions: dict[str, dict]) -> dict[str, dict]:
        """The revision where one succeeded, else the draft; with the cost of reflecting, not of the draft."""
        final = {}
        for qid, c in contexts.items():
            revised = revisions.get(qid)
            use = revised if revised and revised["status"] == "ok" else c.draft
            cost = (critiques.get(qid) or {}).get("cost", 0.0) + (revised or {}).get("cost", 0.0)
            final[qid] = {"question_id": qid, "predicted_answer": use["predicted_answer"],
                          "predicted_evidence": use["predicted_evidence"], "revised": use is revised, "cost": cost}
        return final

    @staticmethod
    def revisions(contexts: dict[str, Context], final: dict[str, dict]) -> list[Revision]:
        """Each question's draft and final answer, in context order."""
        return [Revision(c.question, c.draft["predicted_answer"], final[q]["predicted_answer"]) for q, c in contexts.items()]
