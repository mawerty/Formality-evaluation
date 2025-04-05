from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    """Abstract base class for formality detection models."""
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def predict(self, sentences: List[str]) -> List[float]:
        """Predicts formality scores (0.0 to 1.0)."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_id})"