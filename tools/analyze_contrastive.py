#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" (metrics_*.json).

:
    python tools/analyze_contrastive.py --input experiments/dev_stage1_sanity/logs/metrics_20251031_163925.json         --output work/contrastive_seed42.png

,  metrics_*.json 
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None
    _MATPLOTLIB_AVAILABLE = False


def load_metrics(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected metrics format: {path}")
    return data


def detect_tasks(metrics: List[Dict]) -> List[str]:
    if not metrics:
        return []
    sample = metrics[0]
    prefix = "contrastive_lambda_"
    tasks = [key[len(prefix):] for key in sample.keys() if key.startswith(prefix)]
    return sorted(tasks)


def summarize(metrics: List[Dict], tasks: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not metrics             or not tasks:
        return summary
    last = metrics[-1]
    for task in tasks:
        task_summary = {
            "lambda_last": float(last.get(f"contrastive_lambda_{task}", 0.0)),
            "ratio_last": float(last.get(f"contrastive_ratio_{task}", 0.0)),
            "ratio_target_last": float(last.get(f"contrastive_ratio_target_{task}", 0.0)),
            "effective_mass_last": float(last.get(f"contrastive_effective_mass_{task}", 0.0)),
            "neg_count_last": float(last.get(f"contrastive_neg_count_{task}", 0.0)),
        }
        summary[task] = task_summary
    return summary


def _ensure_output_dir(path: Path) -> None:
    if path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_metrics(metrics: List[Dict], tasks: List[str], output: Path, title: str) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib , .  matplotlib  --no-plot.")

    epochs = [entry.get("epoch", idx) for idx, entry in enumerate(metrics)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes = list(axes)

    def _plot(axis, metric_prefix: str, ylabel: str):
        for task in tasks:
            values = [entry.get(f"{metric_prefix}_{task}", 0.0) for entry in metrics]
            axis.plot(epochs, values, label=task)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle='--', alpha=0.4)

    _plot(axes[0], "contrastive_lambda", " (contrastive weight)")
    _plot(axes[1], "contrastive_ratio", "Contrastive / CE ratio")
    _plot(axes[2], "contrastive_effective_mass", "Effective neg weight mass")

    axes[-1].set_xlabel("Epoch")
    axes[0].legend()
    fig.suptitle(title)

    _ensure_output_dir(output)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def gather_metric_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    metrics = sorted(input_path.glob("metrics_*.json"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze contrastive training metrics")
    parser.add_argument("--input", required=True,
                        help="metrics_*.json ")
    parser.add_argument("--output", default=None,
                        help=" (png/pdf ). ")
    parser.add_argument("--no-plot", action="store_true",
                        help=", ")
    args = parser.parse_args()

    input_path = Path(args.input)
    metric_files = gather_metric_files(input_path)
    if not metric_files:
        raise FileNotFoundError(f" metrics_*.json: {input_path}")

    for metrics_path in metric_files:
        metrics = load_metrics(metrics_path)
        tasks = detect_tasks(metrics)
        summary = summarize(metrics, tasks)
        title = f"{metrics_path.parent.parent.name}/{metrics_path.parent.name}"
        print("\n===", title, "===")
        for task, values in summary.items():
            print(f"  Task={task}")
            for key, val in values.items():
                print(f"    {key}: {val:.4f}")

        if args.no_plot:
            continue
        if not args.output:
            out_name = metrics_path.with_suffix(".png").name
            output_path = metrics_path.parent / out_name
        else:
            output_path = Path(args.output)
            if output_path.is_dir():
                output_path = output_path / (metrics_path.stem + ".png")
        plot_metrics(metrics, tasks, output_path, title)
        print(f"  : {output_path}")


if __name__ == "__main__":
    main()
