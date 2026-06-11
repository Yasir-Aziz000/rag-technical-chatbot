from pathlib import Path
import pdfplumber
from loguru import logger
from src.ingestion.base import BaseDocumentLoader
from src.models.schemas import Document


class PDFLoader(BaseDocumentLoader):
    def load(self) -> list[Document]:
        documents: list[Document] = []

        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                documents.append(
                    Document(
                        content=text.strip(),
                        source=self.source_name,
                        page=page_num,
                        doc_type="pdf",
                    )
                )

        logger.info("Loaded {} pages from '{}'", len(documents), self.source_name)
        return documents
