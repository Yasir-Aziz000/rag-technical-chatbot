from loguru import logger
from src.vectorstore.chroma_store import ChromaVectorStore
from src.models.schemas import RetrievedChunk


class DocumentRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> None:
        self._store = vector_store
        self._k = k
        self._score_threshold = score_threshold

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        chunks = self._store.similarity_search(
            query=query,
            k=self._k,
            score_threshold=self._score_threshold,
        )

        logger.info("Query='{}...' → {} chunks retrieved", query[:60], len(chunks))
        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            page_info = f", page {chunk.page}" if chunk.page else ""
            parts.append(f"[{i}] Source: {chunk.source}{page_info}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)
