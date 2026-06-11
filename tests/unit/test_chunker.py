import pytest
from src.ingestion.chunker import TextChunker
from src.models.schemas import Document


def make_document(text: str, source: str = "test.txt", page: int = 1) -> Document:
    return Document(content=text, source=source, page=page, doc_type="txt")


def test_chunk_single_short_document():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    doc = make_document("Short text.")
    chunks = chunker.chunk([doc])
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."
    assert chunks[0].source == "test.txt"


def test_chunk_long_document_produces_multiple_chunks():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    long_text = "word " * 100  # 500 chars
    doc = make_document(long_text)
    chunks = chunker.chunk([doc])
    assert len(chunks) > 1


def test_chunk_preserves_source_metadata():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    doc = make_document("A " * 100, source="my_doc.pdf", page=3)
    chunks = chunker.chunk([doc])
    for chunk in chunks:
        assert chunk.source == "my_doc.pdf"
        assert chunk.page == 3


def test_chunk_empty_document_returns_empty():
    chunker = TextChunker()
    doc = make_document("")
    chunks = chunker.chunk([doc])
    assert chunks == []


def test_chunk_multiple_documents():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    docs = [
        make_document("Doc one content.", source="a.txt"),
        make_document("Doc two content.", source="b.txt"),
    ]
    chunks = chunker.chunk(docs)
    sources = {c.source for c in chunks}
    assert "a.txt" in sources
    assert "b.txt" in sources


def test_chunk_index_is_sequential():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    doc = make_document("word " * 80, source="test.txt")
    chunks = chunker.chunk([doc])
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
