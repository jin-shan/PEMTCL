#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single Markdown report that aggregates all experiments (Exp1-Exp4)
for both ToxiCN and FG-COLD, including accuracy / precision / recall / F1 in both
weighted and macro variants.

This script reads existing experiment JSON outputs and writes:
  docs/experiment_results_all.md

Paths (WSL):
  /mnt/e/experiments/exp2_uncertainty
  /mnt/e/experiments/exp3
  /mnt/e/experiments/exp4
"""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SEEDS = [41, 42, 43, 44, 45]

EXPERIMENTS_BASE = Path("/mnt/e/experiments")
EXP2 = EXPERIMENTS_BASE / "exp2_uncertainty"
EXP3 = EXPERIMENTS_BASE / "exp3"
EXP4 = EXPERIMENTS_BASE / "exp4"

TASKS = [
    ("toxic", "Toxic"),
    ("toxic_type", "Toxic Type"),
    ("expression", "Expression"),
    ("target", "Target"),
]


def mean_std(values: List[float]) -> Tuple[float, float]:
    return float(stats.mean(values)), float(stats.pstdev(values))


def fmt_ms(ms: Optional[Tuple[float, float]]) -> str:
    if ms is None:
        return ""
    mean, std = ms
    return f"{mean:.5f}  {std:.5f}"


def fmt_pair(w: Optional[Tuple[float, float]], m: Optional[Tuple[float, float]]) -> str:
    if w is None or m is None:
        return ""
    return f"{fmt_ms(w)} / {fmt_ms(m)}"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RunConfig:
    name: str
    run_root: Optional[Path]
    result_relpath: str
    note: str = ""

    def load_seed_metrics(self) -> List[Dict]:
        if self.run_root is None:
            return []
        metrics = []
        for seed in SEEDS:
            p = self.run_root / f"seed_{seed}" / self.result_relpath
            if not p.exists():
                continue
            metrics.append(load_json(p))
        return metrics


def aggregate_keys(metrics_list: List[Dict], keys: List[str]) -> Dict[str, Optional[Tuple[float, float]]]:
    summary: Dict[str, Optional[Tuple[float, float]]] = {}
    for key in keys:
        values = [float(m[key]) for m in metrics_list if key in m]
        summary[key] = mean_std(values) if values else None
    return summary


def task_metrics_table(task: str, configs: List[RunConfig]) -> str:
    keys = [
        f"{task}_accuracy",
        f"{task}_precision",
        f"{task}_precision_macro",
        f"{task}_recall",
        f"{task}_recall_macro",
        f"{task}_f1",
        f"{task}_f1_macro",
    ]

    out: List[str] = []
    out.append("|  | Accuracy | Precisionweighted / macro | Recallweighted / macro | F1weighted / macro |")
    out.append("| --- | --- | --- | --- | --- |")

    for cfg in configs:
        metrics_list = cfg.load_seed_metrics()
        if cfg.run_root is None or not metrics_list:
            acc = ""
            pre = ""
            rec = ""
            f1 = ""
            suffix = "" if cfg.run_root is None else ""
        else:
            summary = aggregate_keys(metrics_list, keys)
            acc = fmt_ms(summary[f"{task}_accuracy"])
            pre = fmt_pair(summary[f"{task}_precision"], summary[f"{task}_precision_macro"])
            rec = fmt_pair(summary[f"{task}_recall"], summary[f"{task}_recall_macro"])
            f1 = fmt_pair(summary[f"{task}_f1"], summary[f"{task}_f1_macro"])
            suffix = ""
        name = cfg.name + (f" {cfg.note}" if cfg.note else "") + (suffix if suffix else "")
        out.append(f"| {name} | {acc} | {pre} | {rec} | {f1} |")

    return "\n".join(out)


def overall_macro_table(configs: List[RunConfig]) -> str:
    keys = ["macro_precision", "macro_recall", "macro_f1"]
    out: List[str] = []
    out.append("|  | macro_precision | macro_recall | macro_f1 |")
    out.append("| --- | --- | --- | --- |")
    for cfg in configs:
        metrics_list = cfg.load_seed_metrics()
        suffix = ""
        if cfg.run_root is None or not metrics_list:
            suffix = "" if cfg.run_root is None else ""
            name = cfg.name + (f" {cfg.note}" if cfg.note else "") + (suffix if suffix else "")
            out.append(f"| {name} |  |  |  |")
            continue

        summary = aggregate_keys(metrics_list, keys)
        name = cfg.name + (f" {cfg.note}" if cfg.note else "")
        out.append(
            f"| {name} | {fmt_ms(summary['macro_precision'])} | {fmt_ms(summary['macro_recall'])} | {fmt_ms(summary['macro_f1'])} |"
        )
    return "\n".join(out)


def ensure_dir(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def build_experiment_1(dataset: str) -> List[RunConfig]:
    base = EXP2 / dataset
    return [
        RunConfig("RoBERTa", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json"),
        RunConfig("BERT", ensure_dir(base / "bert_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json"),
        RunConfig("MacBERT", ensure_dir(base / "macbert_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json"),
    ]


def build_experiment_2_tasks(dataset: str) -> List[RunConfig]:
    base = EXP2 / dataset
    return [
        RunConfig("Full MTL", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json"),
        RunConfig("w/o toxic_type", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix_wo_toxic_type"), "logs/final_results_last.json"),
        RunConfig("w/o expression", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix_wo_expression"), "logs/final_results_last.json"),
        RunConfig("w/o target", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix_wo_target"), "logs/final_results_last.json"),
        RunConfig("toxic-only", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix_toxic_only"), "logs/final_results_last.json"),
    ]


def build_experiment_2_modules(dataset: str) -> List[RunConfig]:
    base = EXP2 / dataset
    no_stage_a = base / "roberta_prompt_stageB_epoch20_nomix_noStageA"
    no_cl = base / "roberta_prompt_stageB_epoch20_noCL"
    return [
        RunConfig("Full", ensure_dir(base / "roberta_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json"),
        RunConfig("No Stage-A", ensure_dir(no_stage_a), "logs/final_results_last.json"),
        RunConfig("No Contrastive", ensure_dir(no_cl), "logs/final_results_last.json"),
    ]


def build_experiment_3_configs() -> List[RunConfig]:
    return [
        RunConfig("ToxiCN  FG-COLD", ensure_dir(EXP3 / "toxicn_to_cold"), "logs/test_results.json"),
        RunConfig("FG-COLD  ToxiCN", ensure_dir(EXP3 / "cold_to_toxicn"), "logs/test_results.json"),
    ]


def build_experiment_4(dataset: str) -> List[RunConfig]:
    base = EXP4 / dataset
    configs: List[RunConfig] = []
    for frac_label, frac_dir in [("10%", "10pct"), ("25%", "25pct"), ("50%", "50pct"), ("75%", "75pct")]:
        configs.append(
            RunConfig(frac_label, ensure_dir(base / frac_dir / "roberta_prompt_stageB_epoch20_nomix"), "logs/final_results_last.json")
        )
    configs.append(
        RunConfig(
            "100% (Exp1 RoBERTa)",
            ensure_dir(EXP2 / dataset / "roberta_prompt_stageB_epoch20_nomix"),
            "logs/final_results_last.json",
        )
    )
    return configs


def write_section(lines: List[str], title: str):
    lines.append("")
    lines.append(f"## {title}")


def write_subsection(lines: List[str], title: str):
    lines.append("")
    lines.append(f"### {title}")


def main() -> None:
    lines: List[str] = []
    lines.append("# ToxiCN & FG-COLD")
    lines.append("")
    lines.append("  5 seeds4145")
    lines.append("")
    lines.append("")
    lines.append("- Accuracy macro ")
    lines.append("- Precision / Recall / F1 weighted macro`*_macro`")
    lines.append("")
    lines.append("")
    lines.append("-  `seed_xx/logs/final_results_last.json`")
    lines.append("- `exp3/*/seed_xx/logs/test_results.json`")
    lines.append("")
    lines.append("> expression  active  `expression_precision/expression_recall/expression_f1`  macro ")

    # Experiment 1
    write_section(lines, "RoBERTa / BERT / MacBERT")
    for dataset in ["toxicn", "fg-cold"]:
        write_subsection(lines, dataset.upper())
        configs = build_experiment_1(dataset)
        lines.append("")
        lines.append("**Overall (macro across tasks)**")
        lines.append(overall_macro_table(configs))
        for task_key, task_name in TASKS:
            lines.append("")
            lines.append(f"**{task_key} ({task_name})**")
            lines.append(task_metrics_table(task_key, configs))

    # Experiment 2
    write_section(lines, "RoBERTa")
    for dataset in ["toxicn", "fg-cold"]:
        write_subsection(lines, f"{dataset.upper()} / ")
        configs = build_experiment_2_tasks(dataset)
        lines.append("")
        lines.append("**Overall (macro across tasks)**")
        lines.append(overall_macro_table(configs))
        for task_key, task_name in TASKS:
            lines.append("")
            lines.append(f"**{task_key} ({task_name})**")
            lines.append(task_metrics_table(task_key, configs))

        write_subsection(lines, f"{dataset.upper()} / ")
        configs = build_experiment_2_modules(dataset)
        lines.append("")
        lines.append("**Overall (macro across tasks)**")
        lines.append(overall_macro_table(configs))
        for task_key, task_name in TASKS:
            lines.append("")
            lines.append(f"**{task_key} ({task_name})**")
            lines.append(task_metrics_table(task_key, configs))

    # Experiment 3
    write_section(lines, "RoBERTa, eval-only")
    configs = build_experiment_3_configs()
    lines.append("")
    lines.append("**Overall (macro across tasks)**")
    lines.append(overall_macro_table(configs))
    for task_key, task_name in TASKS:
        lines.append("")
        lines.append(f"**{task_key} ({task_name})**")
        lines.append(task_metrics_table(task_key, configs))

    # Experiment 4
    write_section(lines, "RoBERTa")
    for dataset in ["toxicn", "fg-cold"]:
        write_subsection(lines, dataset.upper())
        configs = build_experiment_4(dataset)
        lines.append("")
        lines.append("**Overall (macro across tasks)**")
        lines.append(overall_macro_table(configs))
        for task_key, task_name in TASKS:
            lines.append("")
            lines.append(f"**{task_key} ({task_name})**")
            lines.append(task_metrics_table(task_key, configs))

    out_path = Path("docs") / "experiment_results_all.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

