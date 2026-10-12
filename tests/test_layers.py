"""The package layers the README describes: each package imports only the packages below it, and only the CLI prints."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "reflective_qa"
BELOW = {
    "data": set(),
    "retrieval": {"data"},
    "eval": {"data"},
    "runs": set(),
    "agent": {"data", "retrieval", "eval"},
    "experiments": {"data", "retrieval", "eval", "runs", "agent"},
    "site": {"data", "retrieval", "eval", "runs", "agent", "experiments"},
}


def _trees():
    return {path: ast.parse(path.read_text()) for path in SRC.rglob("*.py")}


def test_each_package_imports_only_the_packages_below_it():
    wrong = []
    for path, tree in _trees().items():
        package = path.relative_to(SRC).parts[0]
        allowed = BELOW.get(package, set(BELOW)) | {package, "paths"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reflective_qa."):
                if (imported := node.module.split(".")[1]) not in allowed:
                    wrong.append(f"{path.relative_to(SRC)} imports {imported}")
    assert wrong == []


def test_only_the_cli_prints():
    printing = [str(path.relative_to(SRC)) for path, tree in _trees().items() if path.name != "cli.py"
                and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print" for n in ast.walk(tree))]
    assert printing == []
