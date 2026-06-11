import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "LangChain is a framework for developing LLM applications.\n"
        "It provides abstractions for chains, memory, and agents.\n"
        "ChromaDB is an open-source vector database.\n"
        "It supports similarity search over embedded documents.",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    """Creates a minimal PDF using only stdlib — no extra deps."""
    import struct

    # Minimal single-page PDF with text
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello PDF world) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
trailer<</Size 6/Root 1 0 R>>
startxref
0
%%EOF"""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(content)
    return pdf_path
