import base64

import fitz  # pymupdf
from openai import OpenAI

from .base import HTRModel


def _resize_image(image_base64: str, max_size: int) -> str:
    """Downscale a base64 PNG so its longest edge is at most max_size pixels."""
    img_bytes = base64.b64decode(image_base64)
    pix = fitz.Pixmap(img_bytes)
    if max(pix.width, pix.height) <= max_size:
        return image_base64
    scale = max_size / max(pix.width, pix.height)
    doc = fitz.open("png", img_bytes)
    resized = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()
    return base64.b64encode(resized.tobytes("png")).decode("utf-8")


class LMStudioModel(HTRModel):

    def __init__(self, name: str, model_id: str, base_url: str, max_image_size: int | None = None):
        super().__init__(name, model_id)
        self.max_image_size = max_image_size
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # Dummy key — LMStudio doesn't require auth
            timeout=300.0,        # Local models can be slow on large images
        )

    def transcribe(self, image_base64: str, prompt: str) -> str:
        if self.max_image_size:
            image_base64 = _resize_image(image_base64, self.max_image_size)
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
