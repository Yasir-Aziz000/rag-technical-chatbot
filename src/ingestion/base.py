from abc import ABC, abstractmethod
from pathlib import Path
from src.models.schemas import Document


class BaseDocumentLoader(ABC):
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    @abstractmethod
    def load(self) -> list[Document]:
        """Load file and return a list of Documents, one per logical page/section."""
        ...

    @property
    def source_name(self) -> str:
        return self.file_path.name
