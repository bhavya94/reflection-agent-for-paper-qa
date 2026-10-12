from pathlib import Path

import pytest

from reflective_qa.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "mini_qasper.json"


@pytest.mark.parametrize("command", ["baselines", "reflect"])
def test_paid_commands_refuse_to_start_without_a_budget(command, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # refused before any client exists
    with pytest.raises(SystemExit, match="run needs --max-dollars"):
        main([command, "run", "--dev", str(FIXTURE)])
