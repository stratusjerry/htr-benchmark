import time
from pathlib import Path

from .config import LMSTUDIO_BASE_URL, MODELS, HTR_PROMPT, ModelConfig, load_config
from .evaluate import evaluate
from .models.bedrock import BedrockModel
from .models.gemini import GeminiModel
from .models.lmstudio import LMStudioModel
from .pdf_converter import pdf_to_images


def build_model(model_cfg: ModelConfig, config: dict):
    """Instantiate the right adapter for a ModelConfig."""
    if model_cfg.provider == "gemini":
        if not config.get("gemini_api_key"):
            raise ValueError("GEMINI_API_KEY not set in .env file")
        return GeminiModel(model_cfg.name, model_cfg.model_id, config["gemini_api_key"])
    elif model_cfg.provider == "bedrock":
        if not config.get("bedrock_api_key") or not config.get("bedrock_base_url"):
            raise ValueError("BEDROCK_API_KEY and BEDROCK_BASE_URL must be set in .env file")
        return BedrockModel(
            model_cfg.name, model_cfg.model_id,
            config["bedrock_api_key"], config["bedrock_base_url"],
        )
    elif model_cfg.provider == "lmstudio":
        return LMStudioModel(model_cfg.name, model_cfg.model_id, LMSTUDIO_BASE_URL)
    else:
        raise ValueError(f"Unknown provider: {model_cfg.provider}")


def discover_pdfs(files_dir: Path) -> list[dict]:
    """Find all PDF files in the directory."""
    results = []
    for pdf_path in sorted(files_dir.glob("*.pdf")):
        results.append({
            "pdf_path": pdf_path,
            "ground_truth_path": pdf_path.with_suffix(".txt"),
            "name": pdf_path.stem,
        })
    return results


def discover_samples(files_dir: Path) -> list[dict]:
    """Find PDF files that have matching .txt ground truth files."""
    samples = []
    for entry in discover_pdfs(files_dir):
        if entry["ground_truth_path"].exists():
            samples.append(entry)
        else:
            print(f"  WARNING: No ground truth found for {entry['name']}.pdf, skipping")
    return samples


def _run_gemini_batch(model: GeminiModel, samples: list[dict]) -> list[dict]:
    """Run a Gemini model using batch API across all samples."""
    # Convert all PDFs to images and build a flat list of (sample_idx, page) pairs
    sample_pages = []  # list of (sample_idx, page_images)
    all_images = []    # flat list of base64 images
    page_map = []      # (sample_idx, page_num) for each entry in all_images

    for i, sample in enumerate(samples):
        pages = pdf_to_images(sample["pdf_path"])
        sample_pages.append((i, pages))
        for page_data in pages:
            all_images.append(page_data["base64"])
            page_map.append((i, page_data["page"]))

    print(f"  Submitting {len(all_images)} page(s) across {len(samples)} sample(s)...")
    start = time.time()
    transcriptions = model.transcribe_batch(all_images, HTR_PROMPT)
    elapsed = round(time.time() - start, 2)
    print(f"  Batch completed in {elapsed}s")

    # Group transcriptions back by sample
    sample_texts: dict[int, list[str]] = {}
    for idx, (sample_idx, page_num) in enumerate(page_map):
        sample_texts.setdefault(sample_idx, []).append(transcriptions[idx])

    # Build results
    results = []
    for i, sample in enumerate(samples):
        ground_truth = sample["ground_truth_path"].read_text(encoding="utf-8")
        pages = sample_pages[i][1]
        full_transcription = "\n".join(sample_texts.get(i, []))
        metrics = evaluate(ground_truth, full_transcription)

        results.append({
            "sample": sample["name"],
            "model": model.name,
            "provider": "gemini",
            "transcription": full_transcription,
            "ground_truth": ground_truth,
            "pages": len(pages),
            **metrics,
        })
        print(f"  {sample['name']}: CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}")

    return results


def _run_sequential(model, provider: str, samples: list[dict]) -> list[dict]:
    """Run a model sequentially, one page at a time."""
    total = len(samples)
    results = []

    for sample_num, sample in enumerate(samples, start=1):
        print(f"  [{sample_num}/{total}] {sample['name']}")
        ground_truth = sample["ground_truth_path"].read_text(encoding="utf-8")
        pages = pdf_to_images(sample["pdf_path"])
        all_page_texts = []

        for page_data in pages:
            print(f"    Page {page_data['page']}/{len(pages)}...")
            start = time.time()

            try:
                transcription = model.transcribe(page_data["base64"], HTR_PROMPT)
            except Exception as e:
                transcription = f"[ERROR: {e}]"
                print(f"      ERROR: {e}")

            elapsed = round(time.time() - start, 2)
            all_page_texts.append(transcription)
            print(f"      Done in {elapsed}s ({len(transcription)} chars)")

        full_transcription = "\n".join(all_page_texts)
        metrics = evaluate(ground_truth, full_transcription)

        results.append({
            "sample": sample["name"],
            "model": model.name,
            "provider": provider,
            "transcription": full_transcription,
            "ground_truth": ground_truth,
            "pages": len(pages),
            **metrics,
        })
        print(f"    CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}")

    return results


def _generate_ground_truth_sequential(model, to_process: list[dict]) -> list[Path]:
    """Generate ground truth using standard API (one page at a time)."""
    total = len(to_process)
    generated = []
    for file_num, entry in enumerate(to_process, start=1):
        print(f"  [{file_num}/{total}] {entry['name']}")
        pages = pdf_to_images(entry["pdf_path"])
        all_page_texts = []

        for page_data in pages:
            print(f"    Page {page_data['page']}/{len(pages)}...")
            start = time.time()

            try:
                transcription = model.transcribe(page_data["base64"], HTR_PROMPT)
            except Exception as e:
                print(f"      ERROR: {e}")
                transcription = f"[ERROR: {e}]"

            elapsed = round(time.time() - start, 2)
            all_page_texts.append(transcription)
            print(f"      Done in {elapsed}s ({len(transcription)} chars)")

        full_text = "\n".join(all_page_texts)
        entry["ground_truth_path"].write_text(full_text, encoding="utf-8")
        generated.append(entry["ground_truth_path"])
        print(f"    Saved: {entry['ground_truth_path'].name}")

    return generated


def _generate_ground_truth_batch(model: GeminiModel, to_process: list[dict]) -> list[Path]:
    """Generate ground truth using Gemini batch API."""
    all_images = []
    page_map = []  # (pdf_idx, page_num) for each image

    for i, entry in enumerate(to_process):
        pages = pdf_to_images(entry["pdf_path"])
        for page_data in pages:
            all_images.append(page_data["base64"])
            page_map.append((i, page_data["page"]))

    print(f"  Submitting {len(all_images)} page(s) across {len(to_process)} PDF(s)...")
    start = time.time()
    transcriptions = model.transcribe_batch(all_images, HTR_PROMPT)
    elapsed = round(time.time() - start, 2)
    print(f"  Batch completed in {elapsed}s")

    # Group transcriptions back by PDF
    pdf_texts: dict[int, list[str]] = {}
    for idx, (pdf_idx, _) in enumerate(page_map):
        pdf_texts.setdefault(pdf_idx, []).append(transcriptions[idx])

    generated = []
    for i, entry in enumerate(to_process):
        full_text = "\n".join(pdf_texts.get(i, []))
        entry["ground_truth_path"].write_text(full_text, encoding="utf-8")
        generated.append(entry["ground_truth_path"])
        print(f"  Saved: {entry['ground_truth_path'].name}")

    return generated


def generate_ground_truth(files_dir: Path, batch: bool = False, include: list[str] | None = None) -> list[Path]:
    """Run Gemini on all PDFs and save transcriptions as ground truth .txt files.

    Skips PDFs that already have a .txt file.

    Args:
        files_dir: Directory containing PDF files.
        batch: If True, use Gemini batch API instead of standard API.
        include: Optional list of PDF filenames to process. If None, processes all.

    Returns:
        List of paths to generated .txt files.
    """
    config = load_config()
    if not config.get("gemini_api_key"):
        raise ValueError("GEMINI_API_KEY not set in .env file")

    pdfs = discover_pdfs(files_dir)
    if include is not None:
        include_stems = {Path(f).stem for f in include}
        pdfs = [p for p in pdfs if p["name"] in include_stems]
    if not pdfs:
        print("No PDF files found in", files_dir)
        return []

    to_process = [p for p in pdfs if not p["ground_truth_path"].exists()]
    already_done = len(pdfs) - len(to_process)

    if already_done:
        print(f"  Skipping {already_done} PDF(s) that already have ground truth")

    if not to_process:
        print("All PDFs already have ground truth files.")
        return []

    gemini_cfg = next(m for m in MODELS if m.provider == "gemini")
    model = build_model(gemini_cfg, config)

    mode_label = "batch API" if batch else "standard API"
    print(f"\n{'='*60}")
    print(f"Generating ground truth with: {model.name} ({mode_label})")
    print(f"{'='*60}")

    if batch:
        return _generate_ground_truth_batch(model, to_process)
    else:
        return _generate_ground_truth_sequential(model, to_process)


def run_benchmark(
    files_dir: Path,
    model_names: list[str] | None = None,
    batch: bool = False,
    include: list[str] | None = None,
) -> list[dict]:
    """Run the full benchmark.

    Args:
        files_dir: Directory containing PDFs and .txt ground truth files.
        model_names: Optional list of model names to run. If None, runs all.
        batch: If True, use Gemini batch API instead of standard API.
        include: Optional list of PDF filenames to process. If None, processes all.

    Returns:
        List of result dicts, one per (sample, model) combination.
    """
    config = load_config()
    samples = discover_samples(files_dir)
    if include is not None:
        include_stems = {Path(f).stem for f in include}
        samples = [s for s in samples if s["name"] in include_stems]

    if not samples:
        print("No samples with ground truth found. Exiting.")
        return []

    # Filter models if requested
    models_to_run = MODELS
    if model_names:
        name_set = {n.lower() for n in model_names}
        models_to_run = [m for m in MODELS if m.name.lower() in name_set]
        if not models_to_run:
            print(f"No matching models found for: {model_names}")
            print("Use --list-models to see available models.")
            return []

    results = []

    for model_cfg in models_to_run:
        model = build_model(model_cfg, config)

        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")

        if not model.is_available():
            print(f"  SKIPPED: {model.name} is not available")
            continue

        if batch and isinstance(model, GeminiModel):
            results.extend(_run_gemini_batch(model, samples))
        else:
            results.extend(_run_sequential(model, model_cfg.provider, samples))

    return results
