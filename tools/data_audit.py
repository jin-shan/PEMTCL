#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""/"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime

import torch
import transformers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_processor import analyze_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset diagnostics for ToxiCN")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ToxiCN/data",
        help="Directory containing train.json and test.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/diagnostics",
        help="Directory to write the diagnostics report"
    )
    return parser.parse_args()


def ensure_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = os.path.abspath(args.data_dir)
    train_file = ensure_file(os.path.join(data_dir, "train.json"))
    test_file = ensure_file(os.path.join(data_dir, "test.json"))

    print("=" * 80)
    print("Running dataset audit...")
    print(f"Data directory: {data_dir}")

    train_stats = analyze_dataset(train_file)
    test_stats = analyze_dataset(test_file)

    report = {
        "generated_at": timestamp,
        "data_dir": data_dir,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__
        },
        "train": train_stats,
        "test": test_stats
    }

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"data_audit_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Dataset audit report written to: {output_path}")


if __name__ == "__main__":
    main()
