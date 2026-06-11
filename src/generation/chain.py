from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from src.generation.prompts import rag_prompt
from src.models.schemas import RAGResponse, RetrievedChunk
from src.retrieval.retriever import DocumentRetriever


class RAGChain:
    """
    LCEL-based RAG chain.
    Flow: question → retrieve → format context → prompt → LLM → parse
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        groq_api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._retriever = retriever

        self._llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._chain = rag_prompt | self._llm | StrOutputParser()
        self._model_name = model

    def invoke(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ) -> RAGResponse:
        chunks = self._retriever.retrieve(question)
        context = self._retriever.format_context(chunks)

        answer = self._chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history or [],
            }
        )

        logger.info(
            "Generated answer ({} chars) from {} context chunks",
            len(answer),
            len(chunks),
        )

        return RAGResponse(
            answer=answer,
            sources=chunks,
            model_used=self._model_name,
        )

    def stream(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
    ):
        """Yields answer tokens for streaming UI."""
        chunks = self._retriever.retrieve(question)
        context = self._retriever.format_context(chunks)

        for token in self._chain.stream(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history or [],
            }
        ):
            yield token, chunks
