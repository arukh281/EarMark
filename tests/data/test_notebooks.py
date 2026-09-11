"""The Kaggle and Colab notebooks: format, header cells, and the earmark API they call.

The notebooks cannot run here (they download real corpora), so this checks what can be
checked statically: jupytext percent format, a header cell that states the accelerator,
secrets and output sizes, no hard-coded tokens, and that every ``earmark`` attribute and
call they use exists with those argument names. The data modules and notebooks must also
parse under Python 3.10 rules, since Kaggle and Colab images can lag the local 3.12.
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

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "kaggle_prep_speech.py",
    "kaggle_prep_vctk_musan_noise.py",
    "colab_kokoro_agent_voice.py",
    "colab_libricss_prep.py",
    "kaggle_embeddings.py",
)
HEADER = re.compile(r"\A# ---\n# jupyter:\n(?:#.*\n)*?#       format_name: percent\n(?:#.*\n)*?# ---\n")
TOKEN_PATTERNS = (r"hf_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}")
_MISSING = object()


def _notebook(name: str) -> Path:
    return ROOT / "notebooks" / name


def _earmark_aliases(tree: ast.AST) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "earmark":
            module = importlib.import_module(node.module)
            for alias in node.names:
                value = getattr(module, alias.name, _MISSING)
                if value is _MISSING:
                    value = importlib.import_module(f"{node.module}.{alias.name}")
                aliases[alias.asname or alias.name] = value
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "earmark":
                    aliases[alias.asname or alias.name] = importlib.import_module(alias.name)
    return aliases


def _resolve(node: ast.expr, aliases: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, aliases)
        if base is not None and base is not _MISSING and (inspect.ismodule(base) or inspect.isclass(base)):
            return getattr(base, node.attr, _MISSING)
    return None


def _call_problem(fn: Any, call: ast.Call) -> str | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    starred = any(isinstance(a, ast.Starred) for a in call.args)
    double_star = any(k.arg is None for k in call.keywords)
    kwargs = {k.arg: None for k in call.keywords if k.arg is not None}
    if starred or double_star:
        params = sig.parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            unknown = [k for k in kwargs if k not in params]
            return f"unexpected keywords {unknown}" if unknown else None
        return None
    try:
        sig.bind(*([None] * len(call.args)), **kwargs)
    except TypeError as err:
        return str(err)
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


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_the_jupytext_header_and_an_info_cell(name: str) -> None:
    text = _notebook(name).read_text()
    assert HEADER.match(text), "missing the jupytext percent-format header"
    first_cell = text.split("# %%", 2)[1]
    assert first_cell.startswith(" [markdown]")
    for word in ("Accelerator", "Secrets", "Outputs", "GB"):
        assert word in first_cell, f"the header cell does not state {word}"
    assert "20 GB" in first_cell  # the notebook-output limit is addressed explicitly


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_holds_no_tokens(name: str) -> None:
    text = _notebook(name).read_text()
    for pattern in TOKEN_PATTERNS:
        assert not re.search(pattern, text), f"{name} contains something shaped like a token"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_calls_only_existing_earmark_api(name: str) -> None:
    path = _notebook(name)
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = _earmark_aliases(tree)
    assert aliases, "the notebook should call earmark functions rather than reimplement them"
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _resolve(node, aliases) is _MISSING:
            problems.append(f"line {node.lineno}: {ast.unparse(node)} does not exist")
        if isinstance(node, ast.Call):
            target = _resolve(node.func, aliases)
            if target is not None and target is not _MISSING and callable(target):
                problem = _call_problem(target, node)
                if problem:
                    problems.append(f"line {node.lineno}: {ast.unparse(node.func)}(...): {problem}")
    assert not problems, "\n".join(problems)


SOURCES = sorted((ROOT / "python" / "earmark" / "data").glob("*.py")) + [_notebook(n) for n in NOTEBOOKS]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_parses_under_python_3_10_rules(path: Path) -> None:
    source = path.read_text()
    ast.parse(source, filename=str(path), feature_version=(3, 10))
    assert not _fstring_problems(source), f"f-strings that need Python 3.12 on lines {_fstring_problems(source)}"


def test_the_fstring_check_catches_312_only_syntax() -> None:
    assert _fstring_problems('x = f"{d["k"]}"\n') == [1]
    assert _fstring_problems("x = f\"{'a' + d['k']}\"\n") == []
    assert _fstring_problems('x = f"{chr(10).join(y)}"\n') == []
