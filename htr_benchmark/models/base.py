from abc import ABC, abstractmethod


class HTRModel(ABC):
    """Base class for all HTR model adapters."""

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @abstractmethod
    def transcribe(self, image_base64: str, prompt: str) -> str:
        """Send a base64-encoded image to the model and return transcribed text."""
        ...

    def is_available(self) -> bool:
        """Check if this model is currently reachable."""
        return True
