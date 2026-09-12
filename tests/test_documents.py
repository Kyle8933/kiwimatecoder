from __future__ import annotations

import base64
import json
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

from kiwimatecoder import documents
from kiwimatecoder.tools.read_file import _read_file

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _notebook(cells: list[dict]) -> str:
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": cells,
        }
    )


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------


def test_extract_notebook_renders_cells_and_outputs(tmp_path):
    path = tmp_path / "demo.ipynb"
    path.write_text(
        _notebook(
            [
                {
                    "cell_type": "markdown",
                    "source": ["# Title\n", "some *text*"],
                },
                {
                    "cell_type": "code",
                    "source": ["print('hi')"],
                    "outputs": [
                        {"output_type": "stream", "text": ["hi\n"]},
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": "42",
                                "image/png": base64.b64encode(PNG_1PX).decode(),
                            },
                        },
                    ],
                },
            ]
        )
    )

    content = documents.extract_notebook(path)

    assert "## Cell 1 (markdown)" in content
    assert "# Title" in content
    assert "## Cell 2 (code)" in content
    assert "print('hi')" in content
    assert "Output:" in content
    assert "hi" in content
    assert "42" in content
    assert f"[image/png output, {len(PNG_1PX)} bytes]" in content


def test_extract_notebook_truncates_large_output(tmp_path):
    path = tmp_path / "big.ipynb"
    path.write_text(
        _notebook([{"cell_type": "code", "source": ["x" * (documents.MAX_CHARS + 500)]}])
    )

    content = documents.extract_notebook(path)

    assert "truncated" in content
    assert len(content) < documents.MAX_CHARS + 200


def test_extract_notebook_malformed_json_returns_error(tmp_path):
    path = tmp_path / "broken.ipynb"
    path.write_text("{not json")

    content = documents.extract_notebook(path)

    assert content.startswith("Error:")
    assert "not valid notebook JSON" in content


def test_extract_notebook_missing_cells_returns_error(tmp_path):
    path = tmp_path / "empty.ipynb"
    path.write_text("{}")

    content = documents.extract_notebook(path)

    assert "no 'cells' list" in content


def test_extract_notebook_missing_file_returns_error(tmp_path):
    content = documents.extract_notebook(tmp_path / "ghost.ipynb")
    assert content.startswith("Error:")


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------


def _write_docx(path: Path, document_xml: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


def test_extract_docx_paragraphs_and_entities(tmp_path):
    path = tmp_path / "letter.docx"
    _write_docx(
        path,
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Hello &amp; welcome</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Second</w:t><w:br/><w:t>line</w:t></w:r></w:p>"
            "</w:body></w:document>"
        ),
    )

    content = documents.extract_docx(path)

    assert "Hello & welcome" in content
    assert "Second" in content
    assert "line" in content


def test_extract_docx_invalid_zip_returns_error(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"definitely not a zip")

    content = documents.extract_docx(path)

    assert content.startswith("Error:")
    assert "not a valid docx" in content


def test_extract_docx_missing_document_xml_returns_error(tmp_path):
    path = tmp_path / "partial.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("docProps/app.xml", "<xml/>")

    content = documents.extract_docx(path)

    assert "missing word/document.xml" in content


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


def _handcrafted_pdf(stream: bytes) -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 44 >>\nstream\n" + stream + b"\nendstream\n"
        b"endobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )


def test_extract_pdf_uses_pdftotext_when_available(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(documents.shutil, "which", lambda name: "/usr/bin/pdftotext")
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, b"CLI extracted text\n", b"")

    monkeypatch.setattr(documents.subprocess, "run", fake_run)

    content = documents.extract_pdf(path)

    assert content == "CLI extracted text\n"
    assert captured["args"][0] == "/usr/bin/pdftotext"
    assert captured["args"][1] == str(path)
    assert captured["args"][2] == "-"


def test_extract_pdf_falls_back_to_uncompressed_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    path = tmp_path / "plain.pdf"
    path.write_bytes(_handcrafted_pdf(b"BT /F1 12 Tf (Hello PDF) Tj ET"))

    content = documents.extract_pdf(path)

    assert "Hello PDF" in content


def test_extract_pdf_falls_back_to_flate_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    path = tmp_path / "flate.pdf"
    compressed = zlib.compress(b"BT (Compressed hello) Tj ET")
    path.write_bytes(_handcrafted_pdf(compressed))

    content = documents.extract_pdf(path)

    assert "Compressed hello" in content


def test_extract_pdf_binary_garbage_returns_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    path = tmp_path / "garbage.pdf"
    path.write_bytes(bytes(range(256)) * 8)

    content = documents.extract_pdf(path)

    assert "pdftotext" in content
    assert "poppler" in content


@pytest.mark.parametrize(
    "payload",
    [b"", b"%PDF-1.4", b"\x00\x01\x02", b"stream\nBT (unclosed Tj\nendstream"],
)
def test_extract_pdf_never_raises(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    path = tmp_path / "odd.pdf"
    path.write_bytes(payload)

    content = documents.extract_pdf(path)

    assert isinstance(content, str)


# ---------------------------------------------------------------------------
# Dispatch and read_file integration
# ---------------------------------------------------------------------------


def test_extract_document_dispatches_by_suffix(tmp_path):
    notebook = tmp_path / "a.ipynb"
    notebook.write_text(_notebook([]))
    docx = tmp_path / "b.docx"
    _write_docx(docx, "<w:document><w:body/></w:document>")
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(_handcrafted_pdf(b"BT (hi) Tj ET"))
    text = tmp_path / "d.txt"
    text.write_text("plain")

    assert documents.extract_document(notebook)[1] == "notebook"
    assert documents.extract_document(docx)[1] == "docx"
    assert documents.extract_document(pdf)[1] == "pdf"
    assert documents.extract_document(text) is None


def test_read_file_delegates_notebooks(session):
    path = session.workspace_root / "demo.ipynb"
    path.write_text(_notebook([{"cell_type": "code", "source": ["answer = 42"]}]))

    result = _read_file({"path": "demo.ipynb"}, session)

    assert result.ok
    assert "[notebook document:" in result.content
    assert "answer = 42" in result.content


def test_read_file_documents_note_offset_limit_ignored(session):
    path = session.workspace_root / "demo.ipynb"
    path.write_text(_notebook([{"cell_type": "markdown", "source": ["keep me"]}]))

    result = _read_file({"path": "demo.ipynb", "offset": 5, "limit": 1}, session)

    assert "offset/limit are ignored" in result.content
    assert "keep me" in result.content


def test_read_file_still_rejects_other_binaries(session):
    (session.workspace_root / "b.bin").write_bytes(b"\x00\x01\x02")

    result = _read_file({"path": "b.bin"}, session)

    assert not result.ok
    assert "binary" in result.content


def test_read_file_text_behavior_unchanged(session):
    (session.workspace_root / "a.txt").write_text("hello\nworld\n")

    result = _read_file({"path": "a.txt"}, session)

    assert result.ok
    assert "1\thello" in result.content
    assert "2\tworld" in result.content
