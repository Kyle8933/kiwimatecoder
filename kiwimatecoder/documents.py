"""Pure extractors for notebooks, PDFs, and docx files.

``read_file`` delegates to :func:`extract_document` before its binary check so
structured documents are surfaced as text instead of being rejected. Every
extractor is best-effort and never raises: failures come back as a clear,
model-readable string. Output is capped at :data:`MAX_CHARS` with a truncation
note.

PDF support degrades gracefully: the ``pdftotext`` CLI (poppler) is used when
available, otherwise a conservative stdlib fallback decompresses FlateDecode
streams and pulls text out of ``BT``/``ET`` blocks. If neither yields text, the
result explains how to install poppler.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path
from typing import Any

MAX_CHARS = 200_000
DOCUMENT_SUFFIXES = (".ipynb", ".pdf", ".docx")
PDFTOTEXT_TIMEOUT = 30

PDF_INSTALL_HINT = (
    "Error: no extractable text found in this PDF. Install pdftotext (poppler) "
    "for reliable PDF text extraction, or convert the pages to images and "
    "attach them with view_image."
)


def _truncate(text: str, source: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return (
        text[:MAX_CHARS]
        + f"\n... [truncated: {source} exceeds {MAX_CHARS:,} characters]"
    )


# ---------------------------------------------------------------------------
# Jupyter notebooks (.ipynb)
# ---------------------------------------------------------------------------


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value or "")


def _image_payload_size(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    text = str(value)
    try:
        return len(base64.b64decode(text, validate=False))
    except ValueError:
        return len(text)


def _format_notebook_output(output: dict[str, Any]) -> list[str]:
    output_type = str(output.get("output_type") or "")
    if output_type == "stream":
        text = _join_text(output.get("text")).rstrip("\n")
        return ["Output:", text] if text else []
    if output_type == "error":
        name = str(output.get("ename") or "Error")
        value = str(output.get("evalue") or "")
        return [f"Output error: {name}: {value}".rstrip(": ")]
    data = output.get("data")
    if not isinstance(data, dict):
        return []
    rendered: list[str] = []
    text_plain = data.get("text/plain")
    if text_plain:
        rendered.append(_join_text(text_plain).rstrip("\n"))
    for mime, value in data.items():
        if not isinstance(mime, str) or not mime.startswith("image/"):
            continue
        rendered.append(f"[{mime} output, {_image_payload_size(value)} bytes]")
    return ["Output:", *rendered] if rendered else []


def extract_notebook(path: str | Path) -> str:
    """Render a ``.ipynb`` file as Markdown-ish text with cell outputs."""
    notebook_path = Path(path)
    try:
        raw = notebook_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error: could not read notebook '{notebook_path}': {exc}"
    except UnicodeDecodeError as exc:
        return f"Error: '{notebook_path.name}' is not readable UTF-8 text: {exc}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Error: '{notebook_path.name}' is not valid notebook JSON: {exc}"
    if not isinstance(data, dict):
        return (
            f"Error: '{notebook_path.name}' is not a notebook "
            "(expected a JSON object)."
        )
    cells = data.get("cells")
    if not isinstance(cells, list):
        return f"Error: '{notebook_path.name}' has no 'cells' list."

    lines: list[str] = []
    for index, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            continue
        cell_type = str(cell.get("cell_type") or "unknown")
        lines.append(f"## Cell {index} ({cell_type})")
        source = _join_text(cell.get("source")).rstrip("\n")
        if source:
            lines.append(source)
        for output in cell.get("outputs") or []:
            if isinstance(output, dict):
                lines.extend(_format_notebook_output(output))
    body = "\n".join(lines).rstrip() + "\n" if lines else ""
    return _truncate(body, notebook_path.name)


# ---------------------------------------------------------------------------
# Word documents (.docx)
# ---------------------------------------------------------------------------


def extract_docx(path: str | Path) -> str:
    """Extract text from ``word/document.xml`` inside a ``.docx`` archive."""
    docx_path = Path(path)
    try:
        with zipfile.ZipFile(docx_path) as archive:
            try:
                xml_bytes = archive.read("word/document.xml")
            except KeyError:
                return (
                    f"Error: '{docx_path.name}' is not a valid docx file "
                    "(missing word/document.xml)."
                )
    except zipfile.BadZipFile:
        return f"Error: '{docx_path.name}' is not a valid docx file (not a zip)."
    except OSError as exc:
        return f"Error: could not read docx '{docx_path.name}': {exc}"
    try:
        xml_text = xml_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return f"Error: '{docx_path.name}' has unreadable document.xml: {exc}"

    # Preserve structure before stripping tags: paragraphs and line breaks
    # become newlines, tabs become tabs. Entities are decoded last so text like
    # ``&lt;w:p&gt;`` cannot turn into a fake tag.
    xml_text = re.sub(r"</w:p\s*>", "\n", xml_text)
    xml_text = re.sub(r"<w:br\s*/?>", "\n", xml_text)
    xml_text = re.sub(r"<w:tab\s*/?>", "\t", xml_text)
    text = re.sub(r"<[^>]+>", "", xml_text)
    text = html.unescape(text)
    lines = [line.rstrip() for line in text.splitlines()]
    body = "\n".join(lines).strip()
    return _truncate(body + "\n" if body else "", docx_path.name)


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

_PDF_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_STRING_RE = re.compile(r"\((?:\\.|[^\\()])*\)", re.DOTALL)
_PDF_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}


def _inflate(data: bytes) -> bytes | None:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return None


def _unescape_pdf_string(value: str) -> str:
    result: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char != "\\" or index + 1 >= length:
            result.append(char)
            index += 1
            continue
        nxt = value[index + 1]
        if nxt in "\\()":
            result.append(nxt)
            index += 2
        elif nxt in _PDF_ESCAPES:
            result.append(_PDF_ESCAPES[nxt])
            index += 2
        elif nxt.isdigit():
            digits = nxt
            index += 2
            while index < length and len(digits) < 3 and value[index].isdigit():
                digits += value[index]
                index += 1
            result.append(chr(int(digits, 8) & 0xFF))
        else:
            index += 2
    return "".join(result)


def _extract_pdf_text(stream: bytes) -> str:
    """Pull literal strings out of ``BT``/``ET`` blocks of a content stream."""
    text = stream.decode("latin-1", "replace")
    lines: list[str] = []
    for block in re.findall(r"BT(.*?)ET", text, flags=re.DOTALL):
        pieces = [
            _unescape_pdf_string(match.group(0)[1:-1])
            for match in _PDF_STRING_RE.finditer(block)
        ]
        joined = " ".join(piece for piece in pieces if piece)
        if joined:
            lines.append(joined)
    return "\n".join(lines)


def _pdf_fallback(data: bytes) -> str:
    """Conservative stdlib extraction: inflate streams, then read BT/ET text."""
    chunks: list[str] = []
    for match in _PDF_STREAM_RE.finditer(data):
        raw = match.group(1)
        decoded = _inflate(raw)
        if decoded is None:
            decoded = raw
        chunk = _extract_pdf_text(decoded)
        if chunk:
            chunks.append(chunk)
    return "\n".join(chunks).strip()


def extract_pdf(path: str | Path) -> str:
    """Extract PDF text via ``pdftotext`` when present, else the fallback.

    Never raises; returns :data:`PDF_INSTALL_HINT` when nothing parseable is
    found.
    """
    pdf_path = Path(path)
    executable = shutil.which("pdftotext")
    if executable:
        try:
            completed = subprocess.run(
                # PDF first, "-" as the output file means stdout. (Reversing
                # these would make pdftotext read stdin and overwrite the PDF.)
                [executable, str(pdf_path), "-"],
                capture_output=True,
                timeout=PDFTOTEXT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Error: pdftotext failed: {exc}"
        if completed.returncode == 0:
            stdout = completed.stdout.decode("utf-8", "replace")
            if stdout.strip():
                return _truncate(stdout, pdf_path.name)

    try:
        data = pdf_path.read_bytes()
    except OSError as exc:
        return f"Error: could not read PDF '{pdf_path.name}': {exc}"
    text = _pdf_fallback(data)
    if text.strip():
        return _truncate(text + "\n", pdf_path.name)
    return PDF_INSTALL_HINT


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def extract_document(path: str | Path) -> tuple[str, str] | None:
    """Dispatch by suffix, returning ``(content, kind)`` or None.

    ``kind`` is one of ``"notebook"``, ``"pdf"``, or ``"docx"``. Content may be
    an error string when the file is malformed; callers surface it as-is.
    """
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".ipynb":
        return extract_notebook(document_path), "notebook"
    if suffix == ".pdf":
        return extract_pdf(document_path), "pdf"
    if suffix == ".docx":
        return extract_docx(document_path), "docx"
    return None
