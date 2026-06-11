from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    groq_api_key: str = Field(..., description="Groq API key — get free at console.groq.com")
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.3

    # Vector store
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_collection_name: str = "technical_docs"

    # App
    app_title: str = "RAG Technical Documentation Chatbot"
    max_conversation_history: int = 10


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
