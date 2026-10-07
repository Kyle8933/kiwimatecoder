"""Fixed text with brackets must not be read as Rich markup.

``console.print("Usage: /undo [count]")`` printed "Usage: /undo " because Rich
takes ``[count]`` for a style tag it does not know and drops it. About thirty
usage lines, the ``[y/N]`` of a yes/no question and one ``--help`` text lost
their hints that way. A literal bracket in program text has to be written
``\\[``.

This reads the source and fails on any such text in what is printed (print,
input, status, table cells, panels) and in typer ``help=`` texts. (The
``*_USAGE`` constants are plain text and are printed through ``escape()``.) Values that vary at run time are covered by the other
test_*_markup files, which run the commands.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from rich.style import Style

PACKAGE = Path(__file__).resolve().parent.parent / "kiwimatecoder"
# Rich's own pattern for a tag: a letter, "#", "/" or "@" after the bracket.
TAG = re.compile(r"((\\*)\[([a-z#/@][^[]*?)])")
PRINTING = {"print", "input", "status", "add_row", "Panel"}


def swallowed_tags(text: str) -> list[str]:
    """Bracketed text in ``text`` that Rich would take for a tag and then drop."""
    lost = []
    for match in TAG.finditer(text):
        escapes, tag = match.group(2), match.group(3)
        if len(escapes) % 2 or tag.startswith(("/", "@")):
            continue  # escaped, or a closing tag / link
        try:
            Style.parse(tag)
        except Exception:
            lost.append(match.group(0))
    return lost


def _text_parts(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.lineno, node.value
    elif isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                yield part.lineno, part.value
    elif isinstance(node, ast.BinOp):
        yield from _text_parts(node.left)
        yield from _text_parts(node.right)
    elif isinstance(node, ast.IfExp):
        yield from _text_parts(node.body)
        yield from _text_parts(node.orelse)


def _owner(func: ast.expr) -> str:
    return ast.unparse(func.value) if isinstance(func, ast.Attribute) else ""


def findings(src: str) -> list[tuple[int, list[str]]]:
    found = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            keywords = {kw.arg: kw.value for kw in node.keywords}
            markup = keywords.get("markup")
            if isinstance(markup, ast.Constant) and markup.value is False:
                continue
            candidates: list[ast.expr] = []
            if name in PRINTING and (
                name in ("Panel", "add_row") or "console" in _owner(func).lower()
            ):
                candidates += node.args
                candidates += [v for k, v in keywords.items() if k in ("title", "subtitle")]
            if "help" in keywords:  # typer.Argument / typer.Option
                candidates.append(keywords["help"])
            for arg in candidates:
                for lineno, text in _text_parts(arg):
                    if lost := swallowed_tags(text):
                        found.append((lineno, lost))
    return found


def test_the_check_finds_a_hint_that_rich_would_drop():
    source = 'console.print("[yellow]Usage: /undo [count][/yellow]")\nTable().add_row("[a|b]")\n'

    assert [tags for _line, tags in findings(source)] == [["[count]"], ["[a|b]"]]


def test_the_check_accepts_escaped_brackets_and_real_styles():
    source = (
        'console.print("[yellow]Usage: /undo \\\\[count][/yellow] [bold]x[/bold] [dim green]y[/]")\n'
        'console.print("literal [x]", markup=False)\n'
    )

    assert findings(source) == []


def test_no_fixed_text_in_the_package_loses_its_brackets():
    problems = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for lineno, tags in findings(path.read_text(encoding="utf-8")):
            problems.append(f"{path.relative_to(PACKAGE.parent)}:{lineno}: {tags}")

    assert not problems, "write a literal bracket as \\\\[ :\n" + "\n".join(problems)
