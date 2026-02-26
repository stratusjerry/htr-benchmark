import base64
import time

from google import genai
from google.genai import types

from .base import HTRModel

BATCH_POLL_INTERVAL = 10  # seconds between polling


class GeminiModel(HTRModel):

    def __init__(self, name: str, model_id: str, api_key: str):
        super().__init__(name, model_id)
        self.client = genai.Client(api_key=api_key)

    def transcribe(self, image_base64: str, prompt: str) -> str:
        """Single-image transcription via standard API."""
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

    def transcribe_batch(self, images_base64: list[str], prompt: str) -> list[str]:
        """Submit all images as a batch job, poll until done, return results."""
        requests = []
        for img_b64 in images_base64:
            image_data = base64.b64decode(img_b64)
            requests.append({
                "contents": [
                    types.Content(
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=image_data, mime_type="image/png"),
                        ]
                    )
                ]
            })

        batch_job = self.client.batches.create(
            model=self.model_id,
            src={"inlined_requests": requests},
        )

        print(f"    Batch submitted: {batch_job.name} ({len(requests)} requests)")

        # Poll until done
        while not batch_job.done:
            time.sleep(BATCH_POLL_INTERVAL)
            batch_job = self.client.batches.get(name=batch_job.name)
            print(f"    Batch status: {batch_job.state.name}")

        # Extract results
        results = []
        if batch_job.dest and batch_job.dest.inlined_responses:
            for resp in batch_job.dest.inlined_responses:
                if resp.response and resp.response.text:
                    results.append(resp.response.text.strip())
                elif resp.error:
                    results.append(f"[ERROR: {resp.error}]")
                else:
                    results.append("[ERROR: empty response]")
        else:
            results = [f"[ERROR: batch {batch_job.state.name}]"] * len(images_base64)

        return results

    def is_available(self) -> bool:
        try:
            list(self.client.models.list(config={"page_size": 1}))
            return True
        except Exception:
            return False
