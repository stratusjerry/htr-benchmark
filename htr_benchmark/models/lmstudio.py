from openai import OpenAI

from .base import HTRModel


class LMStudioModel(HTRModel):

    def __init__(self, name: str, model_id: str, base_url: str):
        super().__init__(name, model_id)
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # Dummy key — LMStudio doesn't require auth
            timeout=300.0,        # Local models can be slow on large images
        )

    def transcribe(self, image_base64: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
