"""Run files are named by fingerprints of everything that shapes their requests. A refactor must leave every
fingerprint unchanged, or the code stops finding answers that were already paid for. A deliberate prompt or config
change does change them: update these values on purpose, and expect new run files."""

from reflective_qa.agent.reflection import ReflectionAgent
from reflective_qa.agent.writer import Writer

RUNS = {"sonnet-k20": "b8ad8d4550", "sonnet-closed-book": "12d27ad1ae", "haiku-k5": "67a813f46f"}
SETUPS = {"sonnet-code-feedback": "fa82fcf784", "haiku-code-feedback": "00ee709bbf", "sonnet-self-critique": "7b512f1403",
          "haiku-self-critique": "89f4360277", "haiku-sonnet-critic": "0b1e9b299f",
          "haiku-sonnet-critic-checks": "ef7b912dd9"}


def test_writer_run_fingerprints_are_unchanged():
    assert {name: writer.fingerprint for name, writer in Writer.load_all().items()} == RUNS


def test_reflection_setup_fingerprints_are_unchanged():
    assert {name: agent.fingerprint for name, agent in ReflectionAgent.load_all().items()} == SETUPS
