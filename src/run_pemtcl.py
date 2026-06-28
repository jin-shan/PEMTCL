#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main entrypoint for PEMTCL."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processor import analyze_dataset, create_data_loaders
from evaluator import PEMTCLEvaluator
from logger import PEMTCLLogger
from pemtcl_config import PEMTCLConfig
from pemtcl_model import PEMTCLModel
from trainer import PEMTCLTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEMTCL for fine-grained toxic language detection")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "test"])
    parser.add_argument("--stage", type=str, default="stage_b", choices=["stage_a", "stage_b"])
    parser.add_argument("--model", type=str, default="roberta", choices=["bert", "roberta", "macbert"])
    parser.add_argument("--bert_model", type=str, default="D:/models/bert-base-chinese")
    parser.add_argument("--roberta_model", type=str, default="D:/models/chinese-roberta-wwm-ext")
    parser.add_argument("--macbert_model", type=str, default="D:/models/chinese-macbert-base")
    parser.add_argument("--data_dir", type=str, default="data/toxicn")
    parser.add_argument("--output_dir", type=str, default="pemtcl_results")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt_checkpoint", type=str, default=None)
    parser.add_argument("--analysis_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--prompt_length", type=int, default=16)
    parser.add_argument("--num_source_prompts", type=int, default=4)
    parser.add_argument("--prompt_attn_hidden", type=int, default=256)
    parser.add_argument("--prompt_lr", type=float, default=5e-4)
    parser.add_argument("--prompt_weight_decay", type=float, default=0.01)
    parser.add_argument("--prompt_alpha_init", type=float, default=0.0)
    parser.add_argument("--prompt_alpha_max", type=float, default=0.35)
    parser.add_argument("--prompt_alpha_warmup_epochs", type=int, default=3)
    parser.add_argument("--prompt_entropy_weight", type=float, default=0.015)
    parser.add_argument("--prompt_attn_topk", type=int, default=0)
    parser.add_argument("--prompt_attn_topk_ratio", type=float, default=0.5)
    parser.add_argument("--prompt_attn_threshold", type=float, default=0.05)
    parser.add_argument("--prompt_attn_dropout", type=float, default=0.05)

    parser.add_argument("--contrastive_projection_dim", type=int, default=256)
    parser.add_argument("--contrastive_projection_hidden", type=int, default=0)
    parser.add_argument("--contrastive_dropout", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_momentum", type=float, default=0.999)
    parser.add_argument("--contrastive_queue_size", type=int, default=1024)
    parser.add_argument("--contrastive_topk", type=int, default=32)
    parser.add_argument("--contrastive_min_queue_fraction", type=float, default=0.25)
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=1)
    parser.add_argument("--contrastive_weight_toxic", type=float, default=0.1)
    parser.add_argument("--contrastive_weight_toxic_type", type=float, default=0.1)
    parser.add_argument("--contrastive_weight_expression", type=float, default=0.1)
    parser.add_argument("--contrastive_weight_target", type=float, default=0.1)
    parser.add_argument("--contrastive_target_overlap_threshold", type=float, default=0.4)
    return parser.parse_args()


def resolve_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return str(path)


def build_config(args: argparse.Namespace) -> PEMTCLConfig:
    config = PEMTCLConfig()
    config.update_from_args(args)

    config.model_type = args.model
    config.use_roberta = args.model == "roberta"
    config.bert_model_path = args.bert_model
    config.roberta_model_path = args.roberta_model
    config.macbert_model_path = args.macbert_model
    config.training_stage = args.stage
    config.num_epochs = args.epochs if args.epochs is not None else (10 if args.stage == "stage_a" else 15)
    config.data_dir = resolve_path(args.data_dir)
    config.output_root_dir = resolve_path(args.output_dir)
    config.prompt_checkpoint_path = resolve_path(args.prompt_checkpoint)
    config.analysis_dir = resolve_path(args.analysis_dir)

    config.train_path = str(Path(config.data_dir) / "train.json")
    config.dev_path = str(Path(config.data_dir) / "dev.json")
    config.test_path = str(Path(config.data_dir) / "test.json")
    config.saved_model_dir = str(Path(config.output_root_dir) / "saved_models")
    config.log_dir = str(Path(config.output_root_dir) / "logs")

    config.contrastive_task_weights = {
        "toxic": max(args.contrastive_weight_toxic, 0.0),
        "toxic_type": max(args.contrastive_weight_toxic_type, 0.0),
        "expression": max(args.contrastive_weight_expression, 0.0),
        "target": max(args.contrastive_weight_target, 0.0),
    }
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.mode == "train" and args.stage == "stage_b" and not config.prompt_checkpoint_path:
        raise ValueError("stage_b training requires --prompt_checkpoint from a stage_a run.")

    os.makedirs(config.saved_model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    logger = PEMTCLLogger(config.log_dir, config.project_name)
    logger.logger.info("Running PEMTCL | mode=%s | stage=%s | model=%s", args.mode, config.training_stage, args.model)
    logger.log_config(config.to_dict())

    train_stats = analyze_dataset(config.train_path)
    target_counts = train_stats.get("target_summary", {}).get("per_label_counts", {})
    total_samples = train_stats.get("total_samples", 0)
    if total_samples > 0:
        pos_counts = np.array([float(target_counts.get(str(i), 0)) for i in range(len(config.target_labels))])
        pos_counts = np.maximum(pos_counts, 1.0)
        neg_counts = total_samples - pos_counts
        config.target_pos_weight = (neg_counts / pos_counts).tolist()

    train_loader, val_loader, test_loader = create_data_loaders(config)

    model = PEMTCLModel(config)
    model.to(device)
    model.apply_training_stage(config.training_stage)

    if config.training_stage == "stage_b" and config.prompt_checkpoint_path:
        prompt_state = torch.load(config.prompt_checkpoint_path, map_location="cpu")
        model.load_prompt_state(prompt_state)
        logger.logger.info("Loaded stage_a prompt checkpoint from %s", config.prompt_checkpoint_path)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.logger.info("Total parameters: %s", f"{total_params:,}")
    logger.logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    evaluator = PEMTCLEvaluator(config)
    trainer = PEMTCLTrainer(model, config, logger, evaluator)

    if args.checkpoint:
        checkpoint_path = resolve_path(args.checkpoint)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        trainer.load_checkpoint(checkpoint_path)

    if args.mode == "train":
        trainer.train(train_loader, val_loader, config.num_epochs)
        last_checkpoint = os.path.join(config.saved_model_dir, "last_model.pt")
        if os.path.exists(last_checkpoint):
            trainer.load_model_weights(last_checkpoint)
        metrics = evaluator.evaluate(trainer.model, test_loader, device, mode="test")
        evaluator.save_results(metrics, os.path.join(config.log_dir, "final_results_last.json"))
        summary = trainer.get_training_summary()
        logger.logger.info("Training Summary:")
        for key, value in summary.items():
            logger.logger.info("  %s: %s", key, value)
    else:
        metrics = evaluator.evaluate(model, test_loader, device, mode="test")
        evaluator.save_results(metrics, os.path.join(config.log_dir, "test_results.json"))

    logger.logger.info("Done.")


if __name__ == "__main__":
    main()
