import base64
from pathlib import Path

import fitz  # pymupdf


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[dict]:
    """Convert a PDF to a list of page image dicts.

    Returns:
        List of {"page": int, "image_bytes": bytes, "base64": str}
    """
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72  # 72 is the default PDF DPI
    matrix = fitz.Matrix(zoom, zoom)
    results = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        results.append({
            "page": i,
            "image_bytes": img_bytes,
            "base64": base64.b64encode(img_bytes).decode("utf-8"),
        })
    doc.close()
    return results
