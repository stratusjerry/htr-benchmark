import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    name: str       # Human-readable display name
    model_id: str   # Model identifier for API calls
    provider: str   # "gemini" or "lmstudio"


# Model registry — all supported benchmark targets.
# For LMStudio models, the model_id must match what LMStudio reports.
# Check http://localhost:1234/v1/models to see loaded model identifiers.
MODELS: list[ModelConfig] = [
    ModelConfig(
        name="Google Gemini 2.0 Flash",
        model_id="gemini-2.0-flash",
        provider="gemini",
    ),
    ModelConfig(
        name="Llama 3.2 11B Vision",
        model_id="llama-3.2-11b-vision",
        provider="lmstudio",
    ),
    ModelConfig(
        name="DeepSeek-OCR-GGUF",
        model_id="deepseek-ocr-gguf",
        provider="lmstudio",
    ),
    ModelConfig(
        name="gemma-3-12b",
        model_id="gemma-3-12b",
        provider="lmstudio",
    ),
    ModelConfig(
        name="gemma-3-27b",
        model_id="gemma-3-27b",
        provider="lmstudio",
    ),
    ModelConfig(
        name="olmocr-2-7b",
        model_id="allenai/olmocr-2-7b",
        provider="lmstudio",
    ),
]

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

HTR_PROMPT = (
    "Transcribe all handwritten text visible in this image. "
    "Return only the transcribed text, preserving line breaks. "
    "Do not add commentary, corrections, or formatting."
)


def load_config() -> dict:
    """Load .env and return config dict."""
    load_dotenv()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    }
