import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _make_prefix(results: list[dict]) -> str:
    """Build a filename prefix from the model names in the results."""
    models = sorted({r["model"] for r in results})
    safe = "_".join(m.replace("/", "_").replace(" ", "_") for m in models)
    return safe


def save_csv(results: list[dict], output_dir: Path) -> Path:
    """Save results to a CSV file (metrics only, no full text)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _make_prefix(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{prefix}_{timestamp}.csv"

    fieldnames = [
        "sample", "model", "provider", "pages",
        "cer", "wer",
        "ref_char_count", "hyp_char_count",
        "ref_word_count", "hyp_word_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def save_json(results: list[dict], output_dir: Path) -> Path:
    """Save full results including transcriptions to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _make_prefix(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{prefix}_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return json_path


def save_transcriptions(results: list[dict], output_dir: Path) -> Path:
    """Save individual transcription text files for side-by-side comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _make_prefix(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trans_dir = output_dir / f"transcriptions_{prefix}_{timestamp}"
    trans_dir.mkdir(exist_ok=True)

    for r in results:
        safe_model = r["model"].replace("/", "_").replace(" ", "_")
        filename = f"{r['sample']}__{safe_model}.txt"
        (trans_dir / filename).write_text(r["transcription"], encoding="utf-8")

    return trans_dir


def print_summary_table(results: list[dict]):
    """Print a formatted summary table to the console."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    print(f"{'BENCHMARK RESULTS':^80}")
    print(f"{'='*80}")
    print(f"{'Sample':<20} {'Model':<28} {'CER':>8} {'WER':>8} {'Pages':>6}")
    print(f"{'-'*20} {'-'*28} {'-'*8} {'-'*8} {'-'*6}")

    for r in results:
        cer_str = f"{r['cer']:.4f}" if isinstance(r["cer"], float) else str(r["cer"])
        wer_str = f"{r['wer']:.4f}" if isinstance(r["wer"], float) else str(r["wer"])
        print(f"{r['sample']:<20} {r['model']:<28} {cer_str:>8} {wer_str:>8} {r['pages']:>6}")

    # Per-model averages
    model_metrics = defaultdict(lambda: {"cer_sum": 0.0, "wer_sum": 0.0, "count": 0})
    for r in results:
        if isinstance(r["cer"], float):
            m = model_metrics[r["model"]]
            m["cer_sum"] += r["cer"]
            m["wer_sum"] += r["wer"]
            m["count"] += 1

    print(f"\n{'MODEL AVERAGES':^80}")
    print(f"{'-'*80}")
    print(f"{'Model':<35} {'Avg CER':>10} {'Avg WER':>10} {'Samples':>8}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*8}")
    for model_name, m in model_metrics.items():
        if m["count"] > 0:
            avg_cer = m["cer_sum"] / m["count"]
            avg_wer = m["wer_sum"] / m["count"]
            print(f"{model_name:<35} {avg_cer:>10.4f} {avg_wer:>10.4f} {m['count']:>8}")

    print(f"{'='*80}\n")
