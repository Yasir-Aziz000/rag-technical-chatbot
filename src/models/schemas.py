from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    doc_type: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    chunk_index: int
    doc_id: str


class RetrievedChunk(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RAGResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    model_used: str
