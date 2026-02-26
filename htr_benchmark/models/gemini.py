import base64

from google import genai
from google.genai import types

from .base import HTRModel


class GeminiModel(HTRModel):

    def __init__(self, name: str, model_id: str, api_key: str):
        super().__init__(name, model_id)
        self.client = genai.Client(api_key=api_key)

    def transcribe(self, image_base64: str, prompt: str) -> str:
        image_data = base64.b64decode(image_base64)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    ]
                )
            ],
        )
        return response.text.strip()

    def is_available(self) -> bool:
        try:
            # Lightweight check — list a single model
            list(self.client.models.list(config={"page_size": 1}))
            return True
        except Exception:
            return False
