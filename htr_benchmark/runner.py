import time
from pathlib import Path

from .config import LMSTUDIO_BASE_URL, MODELS, HTR_PROMPT, ModelConfig, load_config
from .evaluate import evaluate
from .models.gemini import GeminiModel
from .models.lmstudio import LMStudioModel
from .pdf_converter import pdf_to_images


def build_model(model_cfg: ModelConfig, gemini_api_key: str | None = None):
    """Instantiate the right adapter for a ModelConfig."""
    if model_cfg.provider == "gemini":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        return GeminiModel(model_cfg.name, model_cfg.model_id, gemini_api_key)
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


def generate_ground_truth(files_dir: Path) -> list[Path]:
    """Run Gemini on all PDFs and save transcriptions as ground truth .txt files.

    Skips PDFs that already have a .txt file.

    Returns:
        List of paths to generated .txt files.
    """
    config = load_config()
    if not config.get("gemini_api_key"):
        raise ValueError("GEMINI_API_KEY not set in .env file")

    pdfs = discover_pdfs(files_dir)
    if not pdfs:
        print("No PDF files found in", files_dir)
        return []

    # Only process PDFs without existing ground truth
    to_process = [p for p in pdfs if not p["ground_truth_path"].exists()]
    already_done = len(pdfs) - len(to_process)

    if already_done:
        print(f"  Skipping {already_done} PDF(s) that already have ground truth")

    if not to_process:
        print("All PDFs already have ground truth files.")
        return []

    gemini_cfg = next(m for m in MODELS if m.provider == "gemini")
    model = build_model(gemini_cfg, config["gemini_api_key"])

    print(f"\n{'='*60}")
    print(f"Generating ground truth with: {model.name}")
    print(f"{'='*60}")

    generated = []
    for entry in to_process:
        pages = pdf_to_images(entry["pdf_path"])
        all_page_texts = []

        for page_data in pages:
            print(f"  Processing {entry['name']} page {page_data['page']}...")
            start = time.time()

            try:
                transcription = model.transcribe(page_data["base64"], HTR_PROMPT)
            except Exception as e:
                print(f"    ERROR: {e}")
                transcription = f"[ERROR: {e}]"

            elapsed = round(time.time() - start, 2)
            all_page_texts.append(transcription)
            print(f"    Done in {elapsed}s ({len(transcription)} chars)")

        full_text = "\n".join(all_page_texts)
        entry["ground_truth_path"].write_text(full_text, encoding="utf-8")
        generated.append(entry["ground_truth_path"])
        print(f"  Saved: {entry['ground_truth_path'].name}")

    return generated


def run_benchmark(
    files_dir: Path,
    model_names: list[str] | None = None,
) -> list[dict]:
    """Run the full benchmark.

    Args:
        files_dir: Directory containing PDFs and .txt ground truth files.
        model_names: Optional list of model names to run. If None, runs all.

    Returns:
        List of result dicts, one per (sample, model) combination.
    """
    config = load_config()
    samples = discover_samples(files_dir)

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
        model = build_model(model_cfg, config.get("gemini_api_key"))

        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")

        if not model.is_available():
            print(f"  SKIPPED: {model.name} is not available")
            continue

        for sample in samples:
            ground_truth = sample["ground_truth_path"].read_text(encoding="utf-8")
            pages = pdf_to_images(sample["pdf_path"])

            all_page_texts = []

            for page_data in pages:
                print(f"  Processing {sample['name']} page {page_data['page']}...")
                start = time.time()

                try:
                    transcription = model.transcribe(page_data["base64"], HTR_PROMPT)
                except Exception as e:
                    transcription = f"[ERROR: {e}]"
                    print(f"    ERROR: {e}")

                elapsed = round(time.time() - start, 2)
                all_page_texts.append(transcription)
                print(f"    Done in {elapsed}s ({len(transcription)} chars)")

            full_transcription = "\n".join(all_page_texts)
            metrics = evaluate(ground_truth, full_transcription)

            results.append({
                "sample": sample["name"],
                "model": model.name,
                "provider": model_cfg.provider,
                "transcription": full_transcription,
                "ground_truth": ground_truth,
                "pages": len(pages),
                **metrics,
            })

            print(f"  {sample['name']}: CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}")

    return results
