import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

from reflective_qa import cli
from reflective_qa.runs.store import RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


def _no_client(**_):
    raise AssertionError("an API client was created")


@pytest.mark.parametrize("command", ["baselines", "reflect"])
def test_paid_commands_refuse_to_start_without_a_budget(command, monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _no_client)
    with pytest.raises(SystemExit, match="run needs --max-dollars"):
        cli.main([command, "run", "--dev", str(FIXTURE)])


def test_a_run_over_budget_is_refused_end_to_end_before_any_batch(monkeypatch, tmp_path):
    def create(**_):
        raise AssertionError("a batch was submitted")
    client = SimpleNamespace(messages=SimpleNamespace(
        count_tokens=lambda **_: SimpleNamespace(input_tokens=1_000_000), batches=SimpleNamespace(create=create)))
    client.with_options = lambda **_: client
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: client)
    monkeypatch.setattr(cli, "RunStore", lambda: RunStore(tmp_path))
    with pytest.raises(SystemExit, match=r"haiku-k5: estimate \$[0-9.]+ is over --max-dollars 0.01"):
        cli.main(["baselines", "run", "haiku-k5", "--dev", str(FIXTURE), "--max-dollars", "0.01"])
    assert list(tmp_path.iterdir()) == []  # no answers and no batch ID were written
