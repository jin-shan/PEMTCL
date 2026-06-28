# -*- coding: utf-8 -*-
"""Configuration for the PEMTCL framework."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class PEMTCLConfig:
    """Runtime configuration for PEMTCL."""

    def __init__(self) -> None:
        self.project_name = "PEMTCL"
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = Path(__file__).resolve().parents[1]
        self.base_dir = str(self.project_root)

        self.data_dir = str(self.project_root / "data" / "toxicn")
        self.train_path = str(Path(self.data_dir) / "train.json")
        self.dev_path = str(Path(self.data_dir) / "dev.json")
        self.test_path = str(Path(self.data_dir) / "test.json")
        self.saved_model_dir = str(self.project_root / "saved_models")
        self.log_dir = str(self.project_root / "logs")
        self.analysis_dir = None
        self.validation_split = 0.1

        self.model_type = "roberta"
        self.bert_model_path = "D:/models/bert-base-chinese"
        self.roberta_model_path = "D:/models/chinese-roberta-wwm-ext"
        self.macbert_model_path = "D:/models/chinese-macbert-base"
        self.use_roberta = True
        self.pad_token_id = 0
        self.vocab_size = None

        self.training_stage = "stage_b"
        self.prompt_checkpoint_path = None

        self.num_epochs = 15
        self.batch_size = 32
        self.learning_rate = 4e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.05
        self.gradient_clip_norm = 1.0
        self.early_stopping_patience = 3
        self.metric_for_best_model = "macro_f1"
        self.eval_steps = 100
        self.save_steps = 500

        self.task_list = ["toxic", "toxic_type", "expression", "target"]
        self.task_weights = {
            "toxic": 0.25,
            "toxic_type": 0.25,
            "expression": 0.25,
            "target": 0.25,
        }
        self.task_uncertainty = True
        self.target_pos_weight = None
        self.expression_class_weight = None

        self.hidden_dropout = 0.1
        self.attention_dropout = 0.1
        self.classifier_dropout = 0.3

        self.toxic_labels = ["Non-toxic", "Toxic"]
        self.toxic_type_labels = ["Offensive", "Hate"]
        self.expression_labels = ["Explicit", "Implicit", "Quoted"]
        self.target_labels = ["LGBTQ", "Region", "Gender", "Race", "Other"]

        self.max_length = 128
        self.truncation = True
        self.padding = "max_length"
        self.return_tensors = "pt"

        self.log_level = "INFO"
        self.log_steps = 50
        self.optimizer_type = "AdamW"
        self.scheduler_type = "linear"

        self.enable_contrastive = True
        self.contrastive_projection_dim = 256
        self.contrastive_projection_hidden = 0
        self.contrastive_dropout = 0.1
        self.contrastive_temperature = 0.07
        self.contrastive_momentum = 0.999
        self.contrastive_queue_size = 1024
        self.contrastive_topk = 32
        self.contrastive_min_queue_fraction = 0.25
        self.contrastive_warmup_epochs = 1
        self.contrastive_task_weights = {
            "toxic": 0.1,
            "toxic_type": 0.1,
            "expression": 0.1,
            "target": 0.1,
        }
        self.contrastive_normalize = True
        self.contrastive_use_momentum = True
        self.contrastive_per_task_queue = True
        self.contrastive_target_overlap_threshold = 0.4
        self.contrastive_multilabel_tasks = ["target", "toxic_type", "expression"]
        self.contrastive_view_dropout = 0.26
        self.contrastive_noise_scale = 0.035
        self.contrastive_use_dual_view = True

        self.use_prompt_mixing = True
        self.prompt_length = 16
        self.num_source_prompts = 4
        self.prompt_attn_hidden = 256
        self.prompt_lr = 5e-4
        self.prompt_weight_decay = 0.01
        self.prompt_alpha_init = 0.0
        self.prompt_alpha_max = 0.35
        self.prompt_alpha_warmup_epochs = 3
        self.prompt_entropy_weight = 0.015
        self.prompt_attn_topk = 0
        self.prompt_attn_topk_ratio = 0.5
        self.prompt_attn_threshold = 0.05
        self.prompt_attn_dropout = 0.05
        self.prompt_collapse_patience = 2
        self.prompt_attn_topk_per_task = {}

    def update_from_args(self, args) -> None:
        for key, value in vars(args).items():
            if value is None:
                continue
            if key == "epochs":
                self.num_epochs = value
            elif key == "model":
                self.model_type = value
            elif key == "stage":
                self.training_stage = value
            elif key == "prompt_checkpoint":
                self.prompt_checkpoint_path = value
            elif hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        data = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.device):
                data[key] = str(value)
            elif isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        return data

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PEMTCLConfig":
        config = cls()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload.items():
            setattr(config, key, value)
        return config
