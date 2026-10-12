"""The public results page: one self-contained HTML file with the collected data embedded. It is published from the
gh-pages branch, which never merges into main."""

import datetime
import json
import subprocess
from pathlib import Path

from reflective_qa.paths import ROOT
from reflective_qa.site.collector import ResultsCollector

TEMPLATE = Path(__file__).with_name("template.html")


def render(data: dict, template: str) -> str:
    """Embed the data in the page. Every "<" is escaped, so text like "</script>" or "<!--" in an answer can't change
    where the script ends."""
    if template.count("__DATA__") != 1:
        raise ValueError("the template needs exactly one __DATA__ placeholder")
    return template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))


def build(collector: ResultsCollector, out: Path) -> Path:
    """Collect the data, stamp it with today's date and the current commit, and write the page."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip() or "unknown"
    data = collector.collect() | {"generated": datetime.date.today().isoformat(), "commit": commit}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, TEMPLATE.read_text()))
    return out
