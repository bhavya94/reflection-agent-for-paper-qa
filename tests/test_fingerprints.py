"""Run files are named by fingerprints of everything that shapes their requests. A refactor must leave every
fingerprint unchanged, or the code stops finding answers that were already paid for. A deliberate prompt or config
change does change them: update these values on purpose, and expect new run files."""

from reflective_qa.answering import load_runs
from reflective_qa.baselines import fingerprint
from reflective_qa.checks import load_max_words
from reflective_qa.reflection import fingerprint as setup_fingerprint, load_setups

RUNS = {"sonnet-k20": "b8ad8d4550", "sonnet-closed-book": "12d27ad1ae", "haiku-k5": "67a813f46f"}
SETUPS = {"sonnet-code-feedback": "fa82fcf784", "haiku-code-feedback": "00ee709bbf", "sonnet-self-critique": "7b512f1403",
          "haiku-self-critique": "89f4360277", "haiku-sonnet-critic": "0b1e9b299f",
          "haiku-sonnet-critic-checks": "ef7b912dd9"}


def test_writer_run_fingerprints_are_unchanged():
    assert {name: fingerprint(run) for name, run in load_runs().items()} == RUNS


def test_reflection_setup_fingerprints_are_unchanged():
    runs, max_words = load_runs(), load_max_words()
    assert {name: setup_fingerprint(s, runs[s.writer], max_words) for name, s in load_setups().items()} == SETUPS
