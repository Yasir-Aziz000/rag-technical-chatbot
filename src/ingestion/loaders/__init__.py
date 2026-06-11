from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .txt_loader import TXTLoader
from pathlib import Path
from src.ingestion.base import BaseDocumentLoader


_REGISTRY: dict[str, type[BaseDocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".txt": TXTLoader,
}


def get_loader(file_path: Path) -> BaseDocumentLoader:
    suffix = file_path.suffix.lower()
    loader_class = _REGISTRY.get(suffix)
    if loader_class is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")
    return loader_class(file_path)


__all__ = ["PDFLoader", "DOCXLoader", "TXTLoader", "get_loader"]
