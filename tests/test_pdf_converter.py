import base64
from pathlib import Path

import pytest

from htr_benchmark.pdf_converter import pdf_to_images


def test_pdf_to_images_nonexistent_file():
    with pytest.raises(Exception):
        pdf_to_images(Path("nonexistent.pdf"))


def test_pdf_to_images_returns_list(tmp_path):
    """Create a minimal PDF and verify conversion produces valid output."""
    import fitz

    # Create a one-page PDF with some text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test handwriting sample")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    results = pdf_to_images(pdf_path)

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert isinstance(results[0]["image_bytes"], bytes)
    assert isinstance(results[0]["base64"], str)
    # Verify the base64 decodes back to the same bytes
    assert base64.b64decode(results[0]["base64"]) == results[0]["image_bytes"]
