"""HTR Benchmark Runner

Usage:
    python run_benchmark.py --generate-ground-truth            # Step 1: generate ground truth via Gemini
    python run_benchmark.py                                    # Step 2: run all models
    python run_benchmark.py --models "gemma-3-12b"             # Run one local model
    python run_benchmark.py --list-models                      # List available models
"""
import argparse
from pathlib import Path

from htr_benchmark.config import MODELS
from htr_benchmark.output import print_summary_table, save_csv, save_json, save_transcriptions
from htr_benchmark.runner import generate_ground_truth, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="HTR Benchmark Runner")
    parser.add_argument(
        "--files-dir", type=Path, default=Path("files"),
        help="Directory containing PDF samples and .txt ground truth",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model names to run (space-separated). Omit to run all.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all configured models and exit",
    )
    parser.add_argument(
        "--generate-ground-truth", action="store_true",
        help="Run Gemini on all PDFs to generate ground truth .txt files, then exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Configured models:")
        for m in MODELS:
            print(f"  - {m.name} ({m.provider})")
        return

    if args.generate_ground_truth:
        generated = generate_ground_truth(args.files_dir)
        if generated:
            print(f"\nGenerated {len(generated)} ground truth file(s).")
        return

    results = run_benchmark(args.files_dir, args.models)

    if results:
        csv_path = save_csv(results, args.output_dir)
        json_path = save_json(results, args.output_dir)
        trans_dir = save_transcriptions(results, args.output_dir)

        print_summary_table(results)

        print(f"CSV saved to:            {csv_path}")
        print(f"JSON saved to:           {json_path}")
        print(f"Transcriptions saved to: {trans_dir}")


if __name__ == "__main__":
    main()
