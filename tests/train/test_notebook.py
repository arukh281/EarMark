"""notebooks/kaggle_train.py and the trainer sources, checked statically.

The notebook needs a Kaggle GPU, so this checks what can be checked here: jupytext
percent format; a header cell stating the accelerator, secrets and output sizes; no
hard-coded tokens; that every ``earmark`` name and call it uses exists with those
arguments; that the command it launches parses; and that it and the trainer modules
parse under Python 3.10 rules, since Kaggle images can lag the local 3.12.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import re
import tokenize
from pathlib import Path
from typing import Any

import pytest

from earmark.train.train import build_parser, resolve_config, train_command

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.py"
HEADER = re.compile(r"\A# ---\n# jupyter:\n(?:#.*\n)*?#       format_name: percent\n(?:#.*\n)*?# ---\n")
TOKEN_PATTERNS = (r"hf_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}")
_MISSING = object()


def _aliases(tree: ast.AST) -> dict[str, Any]:
    names: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "earmark":
            module = importlib.import_module(node.module)
            for alias in node.names:
                names[alias.asname or alias.name] = getattr(module, alias.name, _MISSING)
    return names


def _resolve(node: ast.expr, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, names)
        if base is not None and base is not _MISSING and (inspect.ismodule(base) or inspect.isclass(base)):
            return getattr(base, node.attr, _MISSING)
    return None


def _fstring_problems(source: str) -> list[int]:
    """Lines whose f-strings need Python 3.12 (a reused quote or a backslash in a field)."""
    lines: list[int] = []
    stack: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.FSTRING_START:
            quote = tok.string.lstrip("rRbBfFuU")
            if any(len(q) == 1 and quote[0] == q for q in stack):
                lines.append(tok.start[0])
            stack.append(quote)
        elif tok.type == tokenize.FSTRING_END:
            stack.pop()
        elif tok.type == tokenize.STRING and stack:
            body = tok.string.lstrip("rRbBfFuU")
            if "\\" in body or any(len(q) == 1 and body[0] == q for q in stack):
                lines.append(tok.start[0])
    return lines


def test_notebook_has_the_jupytext_header_and_an_info_cell() -> None:
    text = NOTEBOOK.read_text()
    assert HEADER.match(text)
    first_cell = text.split("# %%", 2)[1]
    assert first_cell.startswith(" [markdown]")
    for phrase in ("Accelerator", "GPU T4 x2", "Secrets", "HF_TOKEN", "Outputs", "20 GB", "P100"):
        assert phrase in first_cell, phrase


def test_notebook_holds_no_tokens() -> None:
    text = NOTEBOOK.read_text()
    for pattern in TOKEN_PATTERNS:
        assert not re.search(pattern, text)


def test_notebook_calls_only_existing_earmark_api() -> None:
    tree = ast.parse(NOTEBOOK.read_text())
    names = _aliases(tree)
    assert {"preset", "default_repo_id", "train_command", "MIN_CUDA_CAPABILITY"} <= set(names)
    problems = [f"{name} does not exist" for name, value in names.items() if value is _MISSING]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _resolve(node.func, names)
            if target is None or target is _MISSING or not callable(target):
                continue
            try:
                inspect.signature(target).bind(*([None] * len(node.args)), **{k.arg: None for k in node.keywords if k.arg})
            except TypeError as err:
                problems.append(f"line {node.lineno}: {ast.unparse(node.func)}(...): {err}")
    assert not problems, problems


def test_the_launched_commands_parse() -> None:
    for extra in ([], ["--max-steps", "20000", "--mixer", '{"p_interferer": 0.7}']):
        cmd = train_command("M-v2", data_root="/kaggle/input", repo_id="someone/earmark-checkpoints",
                            work_dir="/kaggle/working/earmark_logs", extra=extra)  # fmt: skip
        assert cmd[1:3] == ["-m", "earmark.train.train"]
        args = build_parser().parse_args(cmd[3:])
        config = resolve_config(args)
        assert (args.storage, args.device, config.name, config.init_from) == ("hub", "cuda", "M-v2", "M-v1")
    verify = build_parser().parse_args(
        train_command("smoke", data_root="d", repo_id="a/b", work_dir="w", verify=True)[3:]
    )
    assert verify.verify_resume and verify.verify_steps == 100 and verify.verify_tolerance == 1e-3


SOURCES = sorted((ROOT / "python" / "earmark" / "train").glob("*.py")) + [NOTEBOOK]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_parses_under_python_3_10_rules(path: Path) -> None:
    source = path.read_text()
    ast.parse(source, filename=str(path), feature_version=(3, 10))
    assert not _fstring_problems(source)
