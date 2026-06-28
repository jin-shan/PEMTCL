#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metric files across seeds.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing seed_xx subfolders."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed list."
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="logs/final_results.json",
        help="Relative path to metrics file inside each seed directory."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write the aggregated summary JSON."
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=["toxic_f1", "toxic_type_f1", "expression_f1", "target_f1", "macro_f1"],
        help="Metric keys to aggregate."
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(metrics_list: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in keys:
        values = [float(m[key]) for m in metrics_list if key in m]
        if not values:
            continue
        summary[key] = {
            "avg": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values))
        }
    return summary


def main():
    args = parse_args()
    base = Path(args.input_dir).resolve()
    seeds = args.seeds
    metrics_list = []

    for seed in seeds:
        metrics_path = base / f"seed_{seed}" / args.result_file
        if not metrics_path.exists():
            continue
        metrics_list.append(load_metrics(metrics_path))

    summary = aggregate(metrics_list, args.keys)
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
