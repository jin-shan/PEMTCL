#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 seed ?toxic_type ??
?  conda run -n absa_project python tools/diagnose_seed_toxic_type.py ^
    --seed_dir E:/experiments/exp2_uncertainty/toxicn/roberta_prompt_stageB_epoch20_nomix/seed_42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dir", type=str, required=True)
    return p.parse_args()


def load_config(seed_dir: Path) -> Dict[str, Any]:
    logs_dir = seed_dir / "logs"
    cfg_path = logs_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    candidates = sorted(logs_dir.glob("config_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"config.json not found in {logs_dir} (and no config_*.json found)")


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_dir).resolve()

    import sys

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from pemtcl_config import PEMTCLConfig
    from data_processor import create_data_loaders
    from pemtcl_model import PEMTCLModel

    cfg = load_config(seed_dir)
    config = PEMTCLConfig()
    for k, v in cfg.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.saved_model_dir = str(seed_dir / "saved_models")
    config.log_dir = str(seed_dir / "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    _, _, test_loader = create_data_loaders(config, shuffle_train=False)

    model = PEMTCLModel(config).to(device)
    ckpt = torch.load(str(seed_dir / "saved_models" / "last_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    prompt_path = seed_dir / "saved_models" / "last_model_prompt.pt"
    if prompt_path.exists():
        model.load_prompt_state(torch.load(str(prompt_path), map_location="cpu"))

    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            probs = model.get_task_predictions(outputs, apply_softmax=False)["toxic_type"].cpu().numpy()
            max_probs = probs.max(axis=1)
            argmax = probs.argmax(axis=1)
            pred_onehot = np.zeros_like(probs, dtype=int)
            active = max_probs >= 0.5
            if np.any(active):
                idx = np.nonzero(active)[0]
                pred_onehot[idx, argmax[idx]] = 1

            true_onehot = labels["toxic_type"].cpu().numpy().astype(int)

            def to_id(mat: np.ndarray) -> np.ndarray:
                if mat.ndim == 1:
                    mat = mat.reshape(1, -1)
                s = mat.sum(axis=1)
                out = np.zeros(mat.shape[0], dtype=int)
                act = s > 0
                if np.any(act):
                    out[act] = mat[act].argmax(axis=1) + 1
                return out

            pred = to_id(pred_onehot)
            true = to_id(true_onehot)
            all_true.append(true)
            all_pred.append(pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    def _counts(arr):
        u, c = np.unique(arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    print("toxic_type true counts:", _counts(y_true))
    print("toxic_type pred counts:", _counts(y_pred))

    active_labels = [1, 2]
    mask = np.isin(y_true, active_labels)
    yt = y_true[mask]
    yp = y_pred[mask]
    print("active subset size:", int(mask.sum()))
    print("active true counts:", _counts(yt))
    print("active pred counts:", _counts(yp))
    print("confusion (rows true [1,2], cols pred [0,1,2]):")
    print(confusion_matrix(yt, yp, labels=[1, 2], normalize=None))


if __name__ == "__main__":
    main()

