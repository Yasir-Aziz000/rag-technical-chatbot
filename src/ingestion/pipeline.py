from pathlib import Path
import hashlib
from loguru import logger
from src.ingestion.loaders import get_loader
from src.ingestion.chunker import TextChunker
from src.models.schemas import Chunk


class IngestionPipeline:
    """Orchestrates: load → chunk → embed → store."""

    def __init__(self, chunker: TextChunker, vector_store) -> None:
        self._chunker = chunker
        self._vector_store = vector_store

    def ingest(self, file_path: Path) -> int:
        """
        Ingest a single document file.
        Returns the number of chunks stored.
        Skips ingestion if the file has already been processed (content-hash check).
        """
        file_hash = self._file_hash(file_path)

        if self._vector_store.document_exists(file_hash):
            logger.info("'{}' already ingested — skipping", file_path.name)
            return 0

        loader = get_loader(file_path)
        documents = loader.load()

        if not documents:
            logger.warning("No content extracted from '{}'", file_path.name)
            return 0

        chunks = self._chunker.chunk(documents)

        # Attach the file hash so we can check deduplication later
        for chunk in chunks:
            chunk.doc_id = file_hash

        self._vector_store.add_chunks(chunks)
        logger.info("Ingested '{}' → {} chunks stored", file_path.name, len(chunks))
        return len(chunks)

    def ingest_many(self, file_paths: list[Path]) -> dict[str, int]:
        return {fp.name: self.ingest(fp) for fp in file_paths}

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        return hashlib.md5(file_path.read_bytes()).hexdigest()
