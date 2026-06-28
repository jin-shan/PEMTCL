#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize confusion matrices and error stats.")
    parser.add_argument(
        "--summary-dir",
        type=str,
        required=True,
        help="Directory containing aggregated analysis outputs."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()

    expr_path = summary_dir / "expression_confusion_matrix_aggregated.json"
    expr_data = json.loads(expr_path.read_text(encoding="utf-8"))
    labels = expr_data["labels"]
    cm = np.array(expr_data["matrix"], dtype=int)
    total = int(cm.sum())
    print(f"Expression confusion total samples: {total}")
    for idx, label in enumerate(labels):
        row = cm[idx]
        correct = int(row[idx])
        total_row = int(row.sum())
        acc = correct / total_row if total_row else 0.0
        print(f"  Label {idx} -> total={total_row}, accuracy={acc:.3f}")

    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            confusions.append(((i, j), int(cm[i, j])))
    confusions.sort(key=lambda kv: kv[1], reverse=True)
    print("Top-5 expression confusions (true -> pred, count):", confusions[:5])

    tgt_path = summary_dir / "target_label_stats_aggregated.json"
    tgt_stats = json.loads(tgt_path.read_text(encoding="utf-8"))
    print("Target label precision/recall:")
    for item in tgt_stats:
        tp = item["true_positive"]
        fp = item["false_positive"]
        fn = item["false_negative"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {item['label']}: precision={precision:.3f}, recall={recall:.3f}, fp={fp}, fn={fn}")


if __name__ == "__main__":
    main()
