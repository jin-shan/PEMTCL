# -*- coding: utf-8 -*-
"""Dataset loading and preprocessing for PEMTCL."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, BertTokenizerFast


class PEMTCLDataset(Dataset):
    """Standardized dataset reader for ToxiCN and FG-FG-COLD."""

    def __init__(self, data_path: str, config, mode: str = "train") -> None:
        self.data_path = data_path
        self.config = config
        self.mode = mode
        self.data = self._load_data()

        if config.use_roberta:
            self.tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_path, use_fast=True)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_model_path, use_fast=True)

        self._validate_schema()

    def _load_data(self) -> List[Dict]:
        with open(self.data_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _validate_schema(self) -> None:
        if not self.data:
            raise ValueError(f"Dataset is empty: {self.data_path}")
        sample = self.data[0]
        required = ["content", "toxic_one_hot", "toxic_type_one_hot", "expression_one_hot", "target"]
        missing = [field for field in required if field not in sample]
        if missing:
            raise KeyError(f"Missing required fields in {self.data_path}: {missing}")
        if len(sample["toxic_one_hot"]) != 2:
            raise ValueError("toxic_one_hot must have length 2")
        if len(sample["toxic_type_one_hot"]) != 2:
            raise ValueError("toxic_type_one_hot must have length 2")
        if len(sample["expression_one_hot"]) != 3:
            raise ValueError("expression_one_hot must have length 3")
        if len(sample["target"]) != 5:
            raise ValueError("target must have length 5")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        encoded = self.tokenizer(
            str(item["content"]),
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
        )

        toxic_one_hot = [int(value) for value in item["toxic_one_hot"]]
        toxic_type_one_hot = [int(value) for value in item["toxic_type_one_hot"]]
        expression_one_hot = [int(value) for value in item["expression_one_hot"]]
        target = [int(value) for value in item["target"]]

        if sum(toxic_one_hot) != 1:
            raise ValueError(f"Invalid toxic_one_hot label at index {idx}: {toxic_one_hot}")

        labels = {
            "toxic": torch.tensor(int(np.argmax(toxic_one_hot)), dtype=torch.long),
            "toxic_type": torch.tensor(toxic_type_one_hot, dtype=torch.float),
            "expression": torch.tensor(expression_one_hot, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.float),
        }

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0),
            "labels": labels,
            "text": str(item["content"]),
            "original_info": {
                "toxic": int(item.get("toxic", labels["toxic"].item())),
                "toxic_type": item.get("toxic_type", toxic_type_one_hot),
                "expression": item.get("expression", expression_one_hot),
                "target": target,
            },
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
            "labels": {
                "toxic": torch.stack([item["labels"]["toxic"] for item in batch]).long(),
                "toxic_type": torch.stack([item["labels"]["toxic_type"] for item in batch]).float(),
                "expression": torch.stack([item["labels"]["expression"] for item in batch]).float(),
                "target": torch.stack([item["labels"]["target"] for item in batch]).float(),
            },
            "texts": [item["text"] for item in batch],
            "original_info": {
                "toxic": [item["original_info"]["toxic"] for item in batch],
                "toxic_type": [item["original_info"]["toxic_type"] for item in batch],
                "expression": [item["original_info"]["expression"] for item in batch],
                "target": [item["original_info"]["target"] for item in batch],
            },
        }


def create_data_loaders(config, shuffle_train: bool = True):
    full_train_dataset = PEMTCLDataset(config.train_path, config, mode="train")

    if os.path.exists(config.dev_path):
        train_dataset = full_train_dataset
        val_dataset = PEMTCLDataset(config.dev_path, config, mode="dev")
    else:
        total_samples = len(full_train_dataset)
        val_size = max(1, int(total_samples * config.validation_split))
        train_size = total_samples - val_size
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    test_dataset = PEMTCLDataset(config.test_path, config, mode="test")

    config.pad_token_id = full_train_dataset.tokenizer.pad_token_id
    config.vocab_size = full_train_dataset.tokenizer.vocab_size

    common_kwargs = {
        "collate_fn": PEMTCLDataset.collate_fn,
        "num_workers": max(0, int(getattr(config, "num_workers", 4))),
        "pin_memory": bool(getattr(config, "pin_memory", True)),
    }

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle_train, **common_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def analyze_dataset(data_path: str) -> Dict:
    with open(data_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    total_samples = len(data)
    toxic_counter = Counter()
    toxic_type_counter = Counter()
    expression_counter = Counter()
    target_label_counter = Counter()
    target_combo_counter = Counter()
    text_lengths = []

    for item in data:
        text_lengths.append(len(str(item.get("content", ""))))
        toxic_counter[int(item.get("toxic", 0))] += 1
        toxic_type_counter[int(item.get("toxic_type", 0))] += 1
        expression_counter[int(item.get("expression", 0))] += 1

        target = [int(value) for value in item.get("target", [0, 0, 0, 0, 0])]
        target_combo_counter[tuple(target)] += 1
        for idx, value in enumerate(target):
            if value == 1:
                target_label_counter[idx] += 1

    lengths = np.array(text_lengths) if text_lengths else np.array([0])
    zero_target_samples = target_combo_counter.get((0, 0, 0, 0, 0), 0)
    multi_target_samples = sum(1 for combo, count in target_combo_counter.items() if sum(combo) > 1 for _ in range(count))

    return {
        "total_samples": total_samples,
        "toxic_distribution": {str(i): int(toxic_counter.get(i, 0)) for i in range(2)},
        "toxic_type_distribution": {str(i): int(toxic_type_counter.get(i, 0)) for i in range(3)},
        "expression_distribution": {str(i): int(expression_counter.get(i, 0)) for i in range(4)},
        "target_summary": {
            "per_label_counts": {str(i): int(target_label_counter.get(i, 0)) for i in range(5)},
            "no_target_samples": zero_target_samples,
            "no_target_ratio": round(zero_target_samples / total_samples, 6) if total_samples else 0.0,
            "multi_target_samples": multi_target_samples,
            "multi_target_ratio": round(multi_target_samples / total_samples, 6) if total_samples else 0.0,
        },
        "text_length": {
            "average": float(np.mean(lengths)),
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
            "percentile_95": float(np.percentile(lengths, 95)),
        },
    }

