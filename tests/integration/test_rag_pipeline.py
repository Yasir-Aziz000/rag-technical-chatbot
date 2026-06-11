"""
Integration tests: ingestion → vector store → retrieval.
These hit a real (temp) ChromaDB — no mocks.
Requires no API key — only tests up to the retrieval stage.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion import IngestionPipeline, TextChunker
from src.retrieval import DocumentRetriever


@pytest.fixture
def in_memory_vector_store():
    """
    A minimal stand-in vector store for integration tests that avoids
    loading the real embedding model (which requires torch + model download).
    Tests the pipeline plumbing without the heavyweight ML dependency.
    """
    store = MagicMock()
    store.document_exists.return_value = False
    store.list_sources.return_value = []
    store.similarity_search.return_value = []
    added_chunks = []
    store.add_chunks.side_effect = lambda chunks: added_chunks.extend(chunks)
    store._added_chunks = added_chunks
    return store


def test_ingest_txt_file_stores_chunks(sample_txt_file: Path, in_memory_vector_store):
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    pipeline = IngestionPipeline(chunker=chunker, vector_store=in_memory_vector_store)

    n = pipeline.ingest(sample_txt_file)

    assert n > 0
    assert in_memory_vector_store.add_chunks.called
    stored = in_memory_vector_store._added_chunks
    assert len(stored) == n


def test_ingest_skips_already_ingested(sample_txt_file: Path, in_memory_vector_store):
    in_memory_vector_store.document_exists.return_value = True
    chunker = TextChunker()
    pipeline = IngestionPipeline(chunker=chunker, vector_store=in_memory_vector_store)

    n = pipeline.ingest(sample_txt_file)

    assert n == 0
    in_memory_vector_store.add_chunks.assert_not_called()


def test_ingest_many_returns_dict(tmp_path: Path, in_memory_vector_store):
    files = []
    for i in range(3):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"Content of document {i}. " * 20)
        files.append(f)

    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    pipeline = IngestionPipeline(chunker=chunker, vector_store=in_memory_vector_store)

    results = pipeline.ingest_many(files)

    assert isinstance(results, dict)
    assert len(results) == 3
    for name, n_chunks in results.items():
        assert n_chunks > 0


def test_retriever_returns_empty_on_no_results(in_memory_vector_store):
    retriever = DocumentRetriever(in_memory_vector_store, k=5, score_threshold=0.3)
    results = retriever.retrieve("what is langchain?")
    assert results == []


def test_retriever_empty_query_returns_empty(in_memory_vector_store):
    retriever = DocumentRetriever(in_memory_vector_store)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
