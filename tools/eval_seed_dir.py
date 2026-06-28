#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
?seed  final_results_last.json?
?  conda run -n absa_project python tools/eval_seed_dir.py ^
    --seed_dir E:/experiments/exp2_uncertainty/toxicn/roberta_prompt_stageB_epoch20_nomix/seed_42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pemtcl_config import PEMTCLConfig
from data_processor import PEMTCLDataset
from evaluator import PEMTCLEvaluator
from pemtcl_model import PEMTCLModel


def _load_config_from_seed(seed_dir: Path) -> Dict[str, Any]:
    logs_dir = seed_dir / "logs"
    cfg_path = logs_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    #  config_{timestamp}.json?config ?    candidates = sorted(logs_dir.glob("config_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        with candidates[0].open("r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"config.json not found: {cfg_path} (and no config_*.json found in {logs_dir})")


def _apply_config_dict(config: PEMTCLConfig, cfg: Dict[str, Any]) -> None:
    for k, v in cfg.items():
        if not hasattr(config, k):
            continue
        setattr(config, k, v)

    # ?device 
    device_val = getattr(config, "device", None)
    if isinstance(device_val, str):
        config.device = torch.device(device_val if torch.cuda.is_available() else "cpu")


def _load_model(
    config: PEMTCLConfig,
    checkpoint_path: Path,
    prompt_adapter_path: Optional[Path],
    device: torch.device,
) -> PEMTCLModel:
    model = PEMTCLModel(config).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if getattr(config, "use_prompt_mixing", False):
        prompt_state = ckpt.get("prompt_state_dict")
        if prompt_state is None and prompt_adapter_path is not None and prompt_adapter_path.exists():
            prompt_state = torch.load(str(prompt_adapter_path), map_location="cpu")
        if prompt_state is not None:
            model.load_prompt_state(prompt_state)

    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved seed directory and write final_results_last.json")
    p.add_argument("--seed_dir", type=str, required=True, help="Seed directory, e.g. E:/.../seed_42")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path; default: <seed_dir>/saved_models/last_model.pt",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output json path; default: <seed_dir>/logs/final_results_last.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).resolve()
    if not seed_dir.exists():
        raise FileNotFoundError(f"seed_dir not found: {seed_dir}")

    cfg_dict = _load_config_from_seed(seed_dir)
    config = PEMTCLConfig()
    _apply_config_dict(config, cfg_dict)

    #  seed_dir ?config.json 
    config.saved_model_dir = str(seed_dir / "saved_models")
    config.log_dir = str(seed_dir / "logs")

    # Windows ?DataLoader?access violation
    # ?eval_seed_dir ?    config.num_workers = 0
    config.pin_memory = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else (seed_dir / "saved_models" / "last_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    prompt_adapter_path = seed_dir / "saved_models" / "last_model_prompt.pt"
    if not prompt_adapter_path.exists():
        prompt_adapter_path = None

    # ?test_loader train/dev?CPU/IO ?    test_dataset = PEMTCLDataset(config.test_path, config, mode="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(getattr(config, "batch_size", 32)),
        shuffle=False,
        collate_fn=PEMTCLDataset.collate_fn,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=bool(getattr(config, "pin_memory", False)),
    )

    model = _load_model(config, checkpoint_path, prompt_adapter_path, device)
    evaluator = PEMTCLEvaluator(config)

    metrics = evaluator.evaluate(model, test_loader, device, mode="test")

    out_path = Path(args.output).resolve() if args.output else (seed_dir / "logs" / "final_results_last.json")
    os.makedirs(out_path.parent, exist_ok=True)
    evaluator.save_results(metrics, str(out_path))

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

