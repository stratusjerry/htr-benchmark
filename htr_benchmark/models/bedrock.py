from anthropic import Anthropic

from .base import HTRModel


class BedrockModel(HTRModel):

    def __init__(self, name: str, model_id: str, api_key: str, base_url: str):
        super().__init__(name, model_id)
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

    def transcribe(self, image_base64: str, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                    ],
                }
            ],
        )
        return message.content[0].text.strip()

    def is_available(self) -> bool:
        try:
            # Lightweight check — small request
            self.client.messages.create(
                model=self.model_id,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
