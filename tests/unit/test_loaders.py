import pytest
from pathlib import Path
from src.ingestion.loaders import get_loader, TXTLoader
from src.ingestion.loaders.txt_loader import TXTLoader


def test_txt_loader_returns_document(sample_txt_file: Path):
    loader = TXTLoader(sample_txt_file)
    docs = loader.load()
    assert len(docs) == 1
    assert "LangChain" in docs[0].content
    assert docs[0].source == sample_txt_file.name
    assert docs[0].doc_type == "txt"


def test_txt_loader_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    loader = TXTLoader(empty)
    docs = loader.load()
    assert docs == []


def test_get_loader_txt(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    loader = get_loader(f)
    assert isinstance(loader, TXTLoader)


def test_get_loader_unsupported_raises(tmp_path: Path):
    f = tmp_path / "doc.xyz"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_loader(f)


def test_source_name_is_filename(sample_txt_file: Path):
    loader = TXTLoader(sample_txt_file)
    assert loader.source_name == sample_txt_file.name
