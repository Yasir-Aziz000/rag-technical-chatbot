from pathlib import Path
from loguru import logger
from src.ingestion.base import BaseDocumentLoader
from src.models.schemas import Document


class TXTLoader(BaseDocumentLoader):
    def load(self) -> list[Document]:
        text = self.file_path.read_text(encoding="utf-8", errors="replace").strip()

        if not text:
            logger.warning("Empty file: '{}'", self.source_name)
            return []

        documents = [
            Document(
                content=text,
                source=self.source_name,
                page=None,
                doc_type="txt",
            )
        ]

        logger.info("Loaded 1 document from '{}'", self.source_name)
        return documents
