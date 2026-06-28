#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate analysis artifacts across seeds.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing seed_xx subfolders with analysis outputs."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed list to aggregate."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write aggregated summaries."
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics(metrics_list: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    result = {}
    for key in keys:
        values = [float(m[key]) for m in metrics_list if key in m]
        if not values:
            continue
        result[key] = {
            "avg": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values))
        }
    return result


def aggregate_confusion_matrices(seed_dirs: List[Path], output_dir: Path):
    combined = {}
    for run_dir in seed_dirs:
        cm_dir = run_dir / "analysis" / "confusion_matrices"
        if not cm_dir.exists():
            continue
        for cm_file in cm_dir.glob("*_confusion_matrix.json"):
            task = cm_file.stem.replace("_confusion_matrix", "")
            data = load_json(cm_file)
            matrix = np.array(data["matrix"], dtype=int)
            entry = combined.setdefault(
                task,
                {
                    "labels": data["labels"],
                    "matrix": np.zeros_like(matrix, dtype=int)
                }
            )
            if entry["labels"] != data["labels"]:
                raise ValueError(f"Label mismatch detected for task {task}")
            entry["matrix"] += matrix

    for task, data in combined.items():
        output_path = output_dir / f"{task}_confusion_matrix_aggregated.json"
        payload = {
            "labels": data["labels"],
            "matrix": data["matrix"].astype(int).tolist()
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def aggregate_target_stats(seed_dirs: List[Path], output_dir: Path):
    aggregated = {}
    for run_dir in seed_dirs:
        stats_file = run_dir / "analysis" / "target_label_stats.json"
        if not stats_file.exists():
            continue
        stats = load_json(stats_file)
        for item in stats:
            label = item["label"]
            entry = aggregated.setdefault(
                label,
                {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
            )
            for key in entry:
                entry[key] += int(item[key])

    output_path = output_dir / "target_label_stats_aggregated.json"
    payload = [
        {"label": label, **values}
        for label, values in aggregated.items()
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def aggregate_error_samples(seed_dirs: List[Path], output_dir: Path, limit: int = 50):
    toxic_fn = []
    toxic_fp = []
    expression_confusions: Dict[str, List[Dict]] = {}
    target_mismatches = []

    for run_dir in seed_dirs:
        error_file = run_dir / "analysis" / "error_cases" / "error_samples.json"
        if not error_file.exists():
            continue
        errors = load_json(error_file)
        toxic_fn.extend(errors.get("toxic_false_negatives", []))
        toxic_fp.extend(errors.get("toxic_false_positives", []))
        for key, cases in errors.get("expression_confusions", {}).items():
            bucket = expression_confusions.setdefault(key, [])
            bucket.extend(cases)
        target_mismatches.extend(errors.get("target_mismatches", []))

    payload = {
        "toxic_false_negatives": toxic_fn[:limit],
        "toxic_false_positives": toxic_fp[:limit],
        "expression_confusions": {k: v[:limit] for k, v in expression_confusions.items()},
        "target_mismatches": target_mismatches[:limit]
    }

    output_path = output_dir / "error_samples_aggregated.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    base = Path(args.input_dir).resolve()
    seed_dirs = [base / f"seed_{seed}" for seed in args.seeds]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics = []
    for run_dir in seed_dirs:
        metrics_file = run_dir / "logs" / "test_results.json"
        if metrics_file.exists():
            metrics.append(load_json(metrics_file))
    metrics_summary = aggregate_metrics(
        metrics,
        keys=["toxic_f1", "toxic_type_f1", "expression_f1", "target_f1", "macro_f1"]
    )
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    aggregate_confusion_matrices(seed_dirs, output_dir)
    aggregate_target_stats(seed_dirs, output_dir)
    aggregate_error_samples(seed_dirs, output_dir)


if __name__ == "__main__":
    main()
