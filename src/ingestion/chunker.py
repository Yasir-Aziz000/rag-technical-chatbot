from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from loguru import logger
from src.models.schemas import Document, Chunk
import hashlib


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for doc in documents:
            doc_id = hashlib.md5(f"{doc.source}:{doc.page}".encode()).hexdigest()
            lc_docs = self._splitter.create_documents(
                texts=[doc.content],
                metadatas=[{"source": doc.source, "page": doc.page, "doc_id": doc_id}],
            )

            for idx, lc_doc in enumerate(lc_docs):
                chunks.append(
                    Chunk(
                        content=lc_doc.page_content,
                        source=lc_doc.metadata["source"],
                        page=lc_doc.metadata.get("page"),
                        chunk_index=idx,
                        doc_id=doc_id,
                    )
                )

        logger.info(
            "Chunked {} documents into {} chunks",
            len(documents),
            len(chunks),
        )
        return chunks
