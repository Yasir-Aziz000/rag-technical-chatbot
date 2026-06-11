from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from src.models.schemas import Chunk, RetrievedChunk


class ChromaVectorStore:
    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        persist_dir: Path,
        collection_name: str,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        logger.info(
            "ChromaDB ready — collection='{}', dir='{}'",
            collection_name,
            persist_dir,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        lc_docs = [
            LCDocument(
                page_content=chunk.content,
                metadata={
                    "source": chunk.source,
                    "page": chunk.page or 0,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                },
            )
            for chunk in chunks
        ]

        self._store.add_documents(lc_docs)
        logger.debug("Added {} chunks to vector store", len(chunks))

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        results = self._store.similarity_search_with_relevance_scores(query, k=k)

        retrieved = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page") or None,
                score=round(score, 4),
            )
            for doc, score in results
            if score >= score_threshold
        ]

        logger.debug("Retrieved {} chunks for query (threshold={})", len(retrieved), score_threshold)
        return retrieved

    def document_exists(self, doc_id: str) -> bool:
        results = self._store.get(where={"doc_id": {"$eq": doc_id}})
        return bool(results["ids"])

    def list_sources(self) -> list[str]:
        data = self._store.get()
        sources = {meta.get("source", "") for meta in data["metadatas"]}
        return sorted(sources - {""})

    def delete_source(self, source_name: str) -> None:
        self._store.delete(where={"source": {"$eq": source_name}})
        logger.info("Deleted all chunks for source '{}'", source_name)

    def reset(self) -> None:
        self._store.delete_collection()
        logger.warning("Vector store collection deleted")

    @property
    def langchain_retriever(self):
        return self._store.as_retriever
