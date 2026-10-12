"""Build the public results page from the real run files.

    uv run python -m reflective_qa.site --out site/index.html

The page is one self-contained HTML file. It is published from the gh-pages branch, which never merges into main.
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from reflective_qa.data.qasper import QasperSplit
from reflective_qa.paths import DEV_SPLIT, ROOT as PROJECT_ROOT
from reflective_qa.site_data import collect

TEMPLATE = Path(__file__).with_name("site_template.html")


def render(data: dict, template: str) -> str:
    """Embed the data in the page. Every "<" is escaped, so text like "</script>" or "<!--" in an answer can't change
    where the script ends."""
    if template.count("__DATA__") != 1:
        raise ValueError("the template needs exactly one __DATA__ placeholder")
    return template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "site" / "index.html")
    parser.add_argument("--dev", type=Path, default=DEV_SPLIT)
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=PROJECT_ROOT).stdout.strip() or "unknown"
    data = collect(QasperSplit.load(args.dev)) | {"generated": datetime.date.today().isoformat(), "commit": commit}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data, TEMPLATE.read_text()))
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
