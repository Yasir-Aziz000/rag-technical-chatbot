from functools import cached_property
from loguru import logger
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Thin wrapper around HuggingFaceEmbeddings.
    Uses cached_property so the heavy model loads only once per process.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device

    @cached_property
    def embeddings(self) -> HuggingFaceEmbeddings:
        logger.info("Loading embedding model '{}'", self._model_name)
        return HuggingFaceEmbeddings(
            model_name=self._model_name,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": True},
        )
