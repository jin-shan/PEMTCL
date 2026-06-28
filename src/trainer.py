# -*- coding: utf-8 -*-
"""
?"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
import json
import copy
import math
from typing import Dict, Optional, Tuple
from collections import deque

from pemtcl_model import PEMTCLModel


class SimpleWeightNetwork(nn.Module):
    """?.3AHN"""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, features: torch.Tensor, anchor_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        features: (K, D) ?        anchor_label: scalar/1D tensor (? ??(?
        : (K,) 
        """
        if features.size(0) == 0:
            return torch.empty(0, device=features.device)

        logits = self.classifier(features)  # (K, num_classes)
        probs = F.softmax(logits, dim=1)  # (K, num_classes)

        if anchor_label is not None:
            # anchor_label
            if anchor_label.dim() > 0 and anchor_label.numel() > 1:
                # 
                anchor_label_normalized = anchor_label.float()
                if anchor_label_normalized.sum() > 0:
                    anchor_label_normalized = anchor_label_normalized / anchor_label_normalized.sum()
                else:
                    anchor_label_normalized = torch.ones_like(anchor_label_normalized) / anchor_label_normalized.numel()
                # eights = probs @ anchor_label_normalized
                weights = torch.mv(probs, anchor_label_normalized.to(probs.device))  # (K,)
            else:
                if isinstance(anchor_label, torch.Tensor):
                    if anchor_label.dim() == 0:
                        anchor_idx = int(anchor_label.item())
                    else:
                        anchor_idx = int(anchor_label[0].item())
                else:
                    anchor_idx = int(anchor_label)
                anchor_idx = max(0, min(anchor_idx, self.num_classes - 1))
                weights = probs[:, anchor_idx]  # (K,)
        else:
            # nchor
            weights = torch.ones(features.size(0), device=features.device) / self.num_classes

        return weights


class PEMTCLTrainer:
    """"""

    def __init__(self, model, config, logger, evaluator):
        self.model = model
        self.config = config
        self.logger = logger
        self.evaluator = evaluator
        self.device = config.device
        self.task_names = list(self.model.task_classifiers.keys())
        self.contrastive_tasks = getattr(self.model, 'contrastive_tasks', [])
        self.multilabel_tasks = set(getattr(config, 'contrastive_multilabel_tasks', ['target']))
        self.expression_group_tensor = self._build_expression_group_tensor(
            getattr(config, 'contrastive_expression_grouping', {})
        )

        # 
        self.momentum_model = None
        self.neg_weight_networks = nn.ModuleDict()
        self.neg_weight_context_dims = {}
        self.use_contrastive = bool(getattr(config, 'enable_contrastive', False) and self.contrastive_tasks)
        self.contrastive_cfg = {
            'temperature': getattr(config, 'contrastive_temperature', 0.07),
            'momentum': getattr(config, 'contrastive_momentum', 0.999),
            'queue_size': getattr(config, 'contrastive_queue_size', 1024),
            'topk': getattr(config, 'contrastive_topk', 32),
            'min_queue_fraction': getattr(config, 'contrastive_min_queue_fraction', 0.25),
            'warmup_epochs': getattr(config, 'contrastive_warmup_epochs', 1),
            'task_weights': getattr(config, 'contrastive_task_weights', {}),
            'use_momentum': getattr(config, 'contrastive_use_momentum', True),
            'per_task_queue': getattr(config, 'contrastive_per_task_queue', True),
            'target_overlap_threshold': getattr(config, 'contrastive_target_overlap_threshold', 0.4),
            'topk_per_task': getattr(config, 'contrastive_topk_per_task', {}),
        }
        if not isinstance(self.contrastive_cfg['topk_per_task'], dict):
            self.contrastive_cfg['topk_per_task'] = {}
        self.contrastive_state = {}
        self.contrastive_effective_mass_state = {task: 0.0 for task in self.contrastive_tasks}
        self.last_model_state_dict = None
        if self.use_contrastive:
            self._init_momentum_encoder()
            self._init_contrastive_state()
            self._init_negative_weight_networks()
            self.expression_class_weight_tensor = None
            if getattr(config, 'expression_class_weight', None):
                self.expression_class_weight_tensor = torch.tensor(
                    config.expression_class_weight,
                    dtype=torch.float,
                    device=self.device
                )
            self.target_pos_weight_tensor = None
            if getattr(config, 'target_pos_weight', None):
                self.target_pos_weight_tensor = torch.tensor(
                    config.target_pos_weight,
                    dtype=torch.float,
                    device=self.device
                )
        else:
            self.expression_class_weight_tensor = None
            self.target_pos_weight_tensor = None

        self.global_step = 0
        self.current_epoch = 0
        self.training_history = []
        self.prompt_entropy_weight = float(getattr(config, 'prompt_entropy_weight', 0.0))
        self.prompt_alpha_init = float(getattr(config, 'prompt_alpha_init', 0.0))
        self.prompt_alpha_max = float(getattr(config, 'prompt_alpha_max', self.prompt_alpha_init))
        self.prompt_alpha_warmup_epochs = max(1, int(getattr(config, 'prompt_alpha_warmup_epochs', 1)))
        self.current_prompt_alpha = None
        if hasattr(self.model, 'set_prompt_alpha'):
            initial_alpha = self.prompt_alpha_init if getattr(self.model, 'use_prompt_mixing', False) else 1.0
            self.model.set_prompt_alpha(initial_alpha)
            if hasattr(self.model, 'get_prompt_alpha'):
                self.current_prompt_alpha = self.model.get_prompt_alpha()
        self.prompt_collapse_patience = max(1, int(getattr(config, 'prompt_collapse_patience', 2)))
        self.prompt_entropy_collapse_epochs = 0
        self.training_stage = getattr(config, 'training_stage', 'stage_b')

        self.dynamic_weights = {task: config.task_weights.get(task, 1.0) for task in self.task_names}
        self.task_loss_history = {task: [] for task in self.task_names}
        self.model.task_weights = self.dynamic_weights.copy()

        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()

    def _build_expression_group_tensor(self, grouping_cfg: Dict) -> Optional[torch.Tensor]:
        if not grouping_cfg:
            return None
        group_ids = {}
        groups = []
        max_index = max(int(k) for k in grouping_cfg.keys())
        tensor = torch.zeros(max_index + 1, dtype=torch.long)
        next_id = 0
        for key, value in grouping_cfg.items():
            idx = int(key)
            group_name = str(value)
            if group_name not in group_ids:
                group_ids[group_name] = next_id
                next_id += 1
            tensor[idx] = group_ids[group_name]
        return tensor

    def _setup_optimizer(self):
        """."""
        # 
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        backbone_decay = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        if backbone_decay:
            optimizer_grouped_parameters.append({
                'params': backbone_decay,
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate
            })
        backbone_nodecay = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ]
        if backbone_nodecay:
            optimizer_grouped_parameters.append({
                'params': backbone_nodecay,
                'weight_decay': 0.0,
                'lr': self.config.learning_rate
            })
        classifier_params = [p for p in self.model.task_classifiers.parameters() if p.requires_grad]
        if classifier_params:
            optimizer_grouped_parameters.append({
                'params': classifier_params,
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate * 10
            })
        prompt_params = []
        if getattr(self.model, 'use_prompt_mixing', False):
            prompt_params = [param for _, param in self.model.iter_prompt_parameters() if param.requires_grad]
        if prompt_params:
            optimizer_grouped_parameters.append({
                'params': prompt_params,
                'weight_decay': getattr(self.config, 'prompt_weight_decay', self.config.weight_decay),
                'lr': getattr(self.config, 'prompt_lr', self.config.learning_rate * 5)
            })
        if self.use_contrastive and len(self.neg_weight_networks) > 0:
            optimizer_grouped_parameters.append({
                'params': self.neg_weight_networks.parameters(),
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate * 10
            })

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )

    def _init_momentum_encoder(self):
        """Initialize the momentum encoder used by LAHN."""
        if not self.contrastive_cfg['use_momentum']:
            self.momentum_model = None
            return
        self.momentum_model = PEMTCLModel(self.config)
        self.momentum_model.load_state_dict(self.model.state_dict())
        for param in self.momentum_model.parameters():
            param.requires_grad = False
        self.momentum_model.to(self.device)

    def _init_contrastive_state(self):
        """."""
        self.similarity = nn.CosineSimilarity(dim=-1)
        queue_size = int(self.contrastive_cfg['queue_size'])
        per_task = self.contrastive_cfg['per_task_queue']
        self.contrastive_state = {
            'queues': {},
            'labels': {},
            'counts': {}
        }
        if per_task:
            for task in self.contrastive_tasks:
                self.contrastive_state['queues'][task] = deque(maxlen=queue_size)
                self.contrastive_state['labels'][task] = deque(maxlen=queue_size)
                self.contrastive_state['counts'][task] = 0
        else:
            self.contrastive_state['queues']['global'] = deque(maxlen=queue_size)
            self.contrastive_state['labels']['global'] = deque(maxlen=queue_size)
            self.contrastive_state['counts']['global'] = 0

    def _init_negative_weight_networks(self):
        projection_dim = getattr(self.model, 'contrastive_projection_dim', 256)
        dropout = getattr(self.config, 'contrastive_dropout', 0.1)
        self.neg_weight_networks = nn.ModuleDict()
        self.neg_weight_context_dims = {}

        # 1.3: 
        for task in self.contrastive_tasks:
            num_classes = self.model.get_task_classes(task)
            self.neg_weight_networks[task] = SimpleWeightNetwork(
                feature_dim=projection_dim,
                num_classes=num_classes,
                dropout=dropout
            ).to(self.device)

    def _get_queue_key(self, task: str) -> str:
        if self.contrastive_cfg['per_task_queue']:
            return task
        return 'global'

    def _enqueue_samples(self, queue_key: str, features: torch.Tensor, task_labels: torch.Tensor,
                         momentum_features: Optional[torch.Tensor] = None):
        queue = self.contrastive_state['queues'][queue_key]
        label_queue = self.contrastive_state['labels'][queue_key]
        feature_source = momentum_features if momentum_features is not None else features
        feature_cpu = feature_source.detach().cpu()
        label_cpu = task_labels.detach().cpu()

        if feature_cpu.dim() == 1:
            feature_cpu = feature_cpu.unsqueeze(0)

        # abel_cpu?Densor
        # 2D [batch_size, num_targets]
        if queue_key != 'global' and queue_key not in self.multilabel_tasks:
            # ?[batch_size] 
            if label_cpu.dim() == 0:
                label_cpu = label_cpu.unsqueeze(0)
            elif label_cpu.dim() > 1:
                # latten?D
                self.logger.logger.warning(
                    f"[enqueue] Task {queue_key} unexpected label shape: {label_cpu.shape}, flattening to 1D"
                )
                label_cpu = label_cpu.view(-1)
            label_cpu = label_cpu.long()
            # 
            if self.global_step < 5:
                self.logger.logger.info(
                    f"[enqueue-debug] task={queue_key}, batch_labels_shape={label_cpu.shape}, "
                    f"feature_shape={feature_cpu.shape}, sample_label={label_cpu[0].item() if label_cpu.numel() > 0 else None}"
                )
        else:
            # ?[batch_size, num_targets]
            if label_cpu.dim() == 1:
                label_cpu = label_cpu.unsqueeze(0)
            if self.global_step < 5:
                self.logger.logger.info(
                    f"[enqueue-debug] multilabel task={queue_key}, labels_shape={label_cpu.shape}, "
                    f"feature_shape={feature_cpu.shape}"
                )

        for i, (feat, lab) in enumerate(zip(feature_cpu, label_cpu)):
            # 
            if queue_key != 'global' and queue_key not in self.multilabel_tasks:
                if lab.dim() != 0:
                    self.logger.logger.error(
                        f"[enqueue] ERROR: task={queue_key}, sample={i}, lab.shape={lab.shape} (expected scalar), "
                        f"lab={lab.tolist() if lab.numel() < 10 else lab.shape}"
                    )
                    raise ValueError(f"Label dimension error for task {queue_key}: expected scalar, got {lab.shape}")
            queue.append(feat)
            label_queue.append(lab)

        self.contrastive_state['counts'][queue_key] = len(queue)

    def _is_queue_ready(self, queue_key: str) -> bool:
        current_len = self.contrastive_state['counts'].get(queue_key, 0)
        min_fraction = float(self.contrastive_cfg['min_queue_fraction'])
        queue_size = int(self.contrastive_cfg['queue_size'])
        threshold = max(1, int(queue_size * min_fraction))
        return current_len >= threshold

    def _momentum_update(self):
        if not self.use_contrastive or self.momentum_model is None:
            return
        momentum = float(self.contrastive_cfg['momentum'])
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def _gather_momentum_features(self, input_ids, attention_mask, token_type_ids):
        if not self.use_contrastive or self.momentum_model is None:
            return None
        with torch.no_grad():
            momentum_outputs = self.momentum_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )
        return momentum_outputs.get('contrastive_features')

    def _stack_queue(self, queue_key: str):
        queue = self.contrastive_state['queues'][queue_key]
        label_queue = self.contrastive_state['labels'][queue_key]
        if len(queue) == 0:
            return None, None
        features = torch.stack(list(queue)).to(self.device)
        sample_label = label_queue[0]
        if sample_label.dim() == 0:
            labels = torch.stack(list(label_queue)).to(self.device)
        else:
            labels = torch.stack(list(label_queue)).to(self.device)
        return features, labels

    def _prepare_class_labels(self, task: str, labels: torch.Tensor) -> torch.Tensor:
        if task in self.multilabel_tasks:
            return labels.float()
        labels = labels.long()
        num_classes = self.model.get_task_classes(task)
        if labels.numel() > 0:
            min_val = int(labels.min().item())
            max_val = int(labels.max().item())
            if min_val < 0 or max_val >= num_classes:
                raise ValueError(
                    f"[contrastive] {task} labels out of range: [{min_val}, {max_val}] vs num_classes={num_classes}"
                )
        return labels

    def _label_mismatch_mask(self, task: str, anchor_label: torch.Tensor, queue_labels: torch.Tensor):
        if task in self.multilabel_tasks:
            anchor_vec = anchor_label.float()
            queue_vec = queue_labels.float()
            anchor_bool = anchor_vec > 0.5
            queue_bool = queue_vec > 0.5
            intersection = torch.sum(anchor_bool & queue_bool, dim=-1).float()
            union = torch.sum(anchor_bool | queue_bool, dim=-1).float().clamp(min=1.0)
            overlap = intersection / union
            threshold = float(self.contrastive_cfg['target_overlap_threshold'])
            mask = (overlap <= threshold) & (union > 0)
            return mask, overlap
        else:
            anchor_val = int(anchor_label.item())
            queue_labels_flat = queue_labels.long()
            mask = queue_labels_flat != anchor_val
            if task == 'expression' and self.expression_group_tensor is not None \
                    and anchor_val < len(self.expression_group_tensor):
                anchor_group = int(self.expression_group_tensor[anchor_val].item())
                queue_indices_cpu = queue_labels_flat.cpu()
                queue_groups = self.expression_group_tensor[queue_indices_cpu].to(queue_labels_flat.device)
                mask = mask & (queue_groups == anchor_group)
            return mask, None

    def _compute_negative_weights(self, task: str, anchor_feature: torch.Tensor,
                                  neg_features: torch.Tensor, anchor_label: torch.Tensor,
                                  neg_labels: torch.Tensor, overlap: Optional[torch.Tensor],
                                  class_factors: Optional[torch.Tensor] = None):
        """1.3"""
        eps = 1e-6

        if task in self.neg_weight_networks:
            # anchor
            weights = self.neg_weight_networks[task](neg_features, anchor_label)
        else:
            weights = torch.ones(neg_features.size(0), device=neg_features.device)

        if overlap is not None:
            weights = weights * (1.0 - overlap)

        # 
        if class_factors is not None:
            if class_factors.dim() == 0:
                class_factors = class_factors.expand_as(weights)
            weights = weights * class_factors

        # niform_mix 

        return torch.clamp(weights, min=eps)

    def _compute_contrastive_loss(self, features: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                                  task_losses: Dict[str, torch.Tensor], batch_idx: int, input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor]):
        task_loss_log: Dict[str, float] = {}
        state_log: Dict[str, float] = {}
        if not self.use_contrastive or not features:
            return None, task_loss_log, state_log

        if self.current_epoch < int(self.contrastive_cfg['warmup_epochs']):
            return None, task_loss_log, state_log

        momentum_features = self._gather_momentum_features(input_ids, attention_mask, token_type_ids)
        effective_mass_totals = {task: 0.0 for task in self.contrastive_tasks}
        effective_mass_counts = {task: 0 for task in self.contrastive_tasks}
        negative_counts = {task: 0 for task in self.contrastive_tasks}

        loss_terms_total = []
        for task, anchor_features in features.items():
            if task not in self.contrastive_tasks:
                continue
            task_labels = labels.get(task)
            if task_labels is None:
                continue
            task_labels = self._prepare_class_labels(task, task_labels)
            positive_view = None
            if isinstance(anchor_features, dict):
                positive_view = anchor_features.get('positive')
                anchor_features = anchor_features.get('anchor')
            queue_key = self._get_queue_key(task)
            anchor_detached = anchor_features.detach()
            momentum_task_features = None
            if momentum_features and task in momentum_features:
                momentum_task_features = momentum_features[task]
                if isinstance(momentum_task_features, dict):
                    momentum_task_features = momentum_task_features.get('anchor')

            self._enqueue_samples(queue_key, anchor_detached, task_labels, momentum_task_features)

            queue_features, queue_labels = self._stack_queue(queue_key)
            if queue_features is None:
                continue
            queue_labels = self._prepare_class_labels(task, queue_labels)

            queue_fill = self.contrastive_state['counts'][queue_key] / float(
                self.contrastive_state['queues'][queue_key].maxlen or 1
            )
            state_log[f'queue_fill_{task}'] = queue_fill

            if not self._is_queue_ready(queue_key):
                continue

            lambda_task = float(self.contrastive_cfg['task_weights'].get(task, 0.0))
            if lambda_task <= 0:
                continue

            if task in self.multilabel_tasks:
                anchor_labels_tensor = task_labels.float()
            else:
                anchor_labels_tensor = task_labels.long()

            if momentum_task_features is not None:
                positive_features = momentum_task_features.to(self.device)
            elif positive_view is not None:
                positive_features = positive_view
            else:
                positive_features = anchor_features.detach()

            anchor_raw = anchor_features
            if getattr(self.config, 'contrastive_normalize', True):
                anchor_features_norm = F.normalize(anchor_features, dim=-1)
                positive_features_norm = F.normalize(positive_features, dim=-1)
                queue_features_norm = F.normalize(queue_features, dim=-1)
            else:
                anchor_features_norm = anchor_features
                positive_features_norm = positive_features
                queue_features_norm = queue_features

            temperature = float(self.contrastive_cfg['temperature'])
            sample_losses = []
            neg_weight_means = []
            neg_sim_means = []

            for idx in range(anchor_features_norm.size(0)):
                anchor_vector = anchor_features_norm[idx]
                positive_vector = positive_features_norm[idx]
                anchor_label = anchor_labels_tensor[idx]
                if batch_idx == 0 and idx == 0 and self.global_step == 0:
                    self.logger.logger.info(
                        f"[contrastive-debug] task={task}, anchor_label={anchor_label.detach().cpu().tolist()}, "
                        f"queue_labels_shape={queue_labels.shape}, queue_sample="
                        f"{queue_labels[: min(5, queue_labels.size(0))].detach().cpu().tolist()}"
                    )

                mask, overlap = self._label_mismatch_mask(task, anchor_label, queue_labels)
                if mask is None or mask.sum() == 0:
                    continue
                valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    continue

                similarities = torch.mv(queue_features_norm[valid_indices], anchor_vector)
                class_factors = None
                if task == 'expression' and self.expression_class_weight_tensor is not None:
                    expr_indices = torch.clamp(
                        queue_labels[valid_indices].detach().cpu().long(),
                        0,
                        self.expression_class_weight_tensor.numel() - 1
                    )
                    class_factors = self.expression_class_weight_tensor[expr_indices].to(queue_features_norm.device)
                elif task == 'target' and self.target_pos_weight_tensor is not None:
                    label_weights = torch.matmul(queue_labels[valid_indices].float(),
                                                 self.target_pos_weight_tensor.to(queue_features_norm.device))
                    class_factors = 1.0 + label_weights / self.target_pos_weight_tensor.sum().clamp_min(1.0)
                negative_weights_all = self._compute_negative_weights(
                    task,
                    anchor_raw[idx],
                    queue_features_norm[valid_indices],
                    anchor_label,
                    queue_labels[valid_indices],
                    overlap[valid_indices] if overlap is not None else None,
                    class_factors
                )

                # 
                weighted_similarities = similarities * negative_weights_all

                task_topk = int(self.contrastive_cfg.get('topk_per_task', {}).get(
                    task,
                    self.contrastive_cfg.get('topk', 0)
                ))
                if task_topk <= 0:
                    task_topk = self.contrastive_cfg.get('topk', 1)
                topk = min(task_topk, valid_indices.numel())
                top_values, top_order = torch.topk(weighted_similarities, topk)
                selected_indices = valid_indices[top_order]

                negative_features = queue_features_norm[selected_indices]
                negative_labels = queue_labels[selected_indices]
                overlap_selected = overlap[selected_indices] if overlap is not None else None

                negative_weights = negative_weights_all[top_order]
                effective_mass = float(negative_weights.sum().item())
                effective_mass_totals[task] = effective_mass_totals.get(task, 0.0) + effective_mass
                effective_mass_counts[task] = effective_mass_counts.get(task, 0) + 1
                negative_counts[task] = negative_counts.get(task, 0) + negative_weights.numel()

                pos_sim = torch.dot(anchor_vector, positive_vector) / temperature
                neg_sims = torch.mv(negative_features, anchor_vector) / temperature

                exp_pos = torch.exp(pos_sim)
                exp_neg = torch.exp(neg_sims) * negative_weights
                loss_sample = -torch.log(exp_pos / (exp_pos + exp_neg.sum()))
                sample_losses.append(loss_sample)

                neg_weight_means.append(negative_weights.mean().detach())
                neg_sim_means.append((neg_sims.detach() * temperature).mean())

            if not sample_losses:
                continue

            task_contrastive_loss = torch.stack(sample_losses).mean()
            task_loss_log[task] = task_contrastive_loss.detach().item()

            avg_mass = 0.0
            avg_negatives = 0.0
            if effective_mass_counts.get(task, 0) > 0:
                avg_mass = effective_mass_totals.get(task, 0.0) / effective_mass_counts[task]
                avg_negatives = negative_counts.get(task, 0) / effective_mass_counts[task]
            state_log[f'contrastive_effective_mass_{task}'] = avg_mass
            state_log[f'contrastive_neg_count_{task}'] = avg_negatives
            self.contrastive_effective_mass_state[task] = avg_mass
            state_log[f'contrastive_lambda_{task}'] = float(lambda_task)
            loss_terms_total.append(float(lambda_task) * task_contrastive_loss)

            if neg_weight_means:
                state_log[f'neg_weight_{task}'] = torch.stack(neg_weight_means).mean().item()
            if neg_sim_means:
                state_log[f'neg_sim_{task}'] = torch.stack(neg_sim_means).mean().item()

        if not loss_terms_total:
            return None, task_loss_log, state_log

        total_contrastive_loss = torch.stack(loss_terms_total).sum()
        return total_contrastive_loss, task_loss_log, state_log

    def setup_scheduler(self, num_training_steps):
        """"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )

    def _apply_prompt_alpha_schedule(self):
        """epoch prompt """
        if not getattr(self.model, 'use_prompt_mixing', False):
            self.current_prompt_alpha = None
            return None
        init_alpha = float(getattr(self.config, 'prompt_alpha_init', 0.0))
        max_alpha = float(getattr(self.config, 'prompt_alpha_max', init_alpha))
        warmup_epochs = max(1, int(getattr(self.config, 'prompt_alpha_warmup_epochs', 1)))
        if max_alpha <= init_alpha:
            alpha = max_alpha
        else:
            progress = min(1.0, max(0.0, self.current_epoch / warmup_epochs))
            alpha = init_alpha + (max_alpha - init_alpha) * progress
        if hasattr(self.model, 'set_prompt_alpha'):
            self.model.set_prompt_alpha(alpha)
        self.current_prompt_alpha = alpha
        return alpha

    def _maybe_adjust_prompt_dropout(self, epoch_metrics: Dict[str, float]):
        """romptdropout"""
        if not getattr(self.model, 'use_prompt_mixing', False):
            self.prompt_entropy_collapse_epochs = 0
            return
        entropy = epoch_metrics.get('prompt_entropy_mean')
        if entropy is None:
            self.prompt_entropy_collapse_epochs = 0
            return
        collapse_threshold = 1e-4
        if entropy <= collapse_threshold:
            self.prompt_entropy_collapse_epochs += 1
        else:
            self.prompt_entropy_collapse_epochs = 0
        if self.prompt_entropy_collapse_epochs >= self.prompt_collapse_patience:
            current_dropout = float(getattr(self.model, 'prompt_attn_dropout_prob', 0.0))
            new_dropout = min(0.9, current_dropout + 0.05)
            if hasattr(self.model, 'set_prompt_dropout') and new_dropout - current_dropout > 1e-4:
                self.model.set_prompt_dropout(new_dropout)
                self.logger.logger.warning(
                    "[prompt-mixing] entropy %.6f indicates collapse. Dropout increased to %.2f",
                    entropy,
                    new_dropout
                )
            self.prompt_entropy_collapse_epochs = 0

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """poch"""
        self.model.train()
        if self.use_contrastive and self.momentum_model is not None:
            self.momentum_model.eval()
        epoch_loss = 0
        epoch_task_losses = {task: 0 for task in self.task_names}
        epoch_contrastive_losses = {task: 0 for task in self.contrastive_tasks}
        contrastive_counts = {task: 0 for task in self.contrastive_tasks}
        contrastive_state_sums: Dict[str, float] = {}
        contrastive_state_counts: Dict[str, int] = {}
        prompt_stat_sums: Dict[str, float] = {}
        prompt_stat_counts: Dict[str, int] = {}
        prompt_entropy_values = []
        prompt_entropy_penalties = []
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            # 
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs['total_loss']
            task_losses = outputs['task_losses']
            contrastive_features = outputs.get('contrastive_features')
            prompt_stats = outputs.get('prompt_stats')
            if prompt_stats:
                for task, stats in prompt_stats.items():
                    for key, value in stats.items():
                        metric_key = f'prompt_{task}_{key}'
                        prompt_stat_sums[metric_key] = prompt_stat_sums.get(metric_key, 0.0) + float(value)
                        prompt_stat_counts[metric_key] = prompt_stat_counts.get(metric_key, 0) + 1
            prompt_entropy = outputs.get('prompt_entropy')
            if prompt_entropy is not None:
                prompt_entropy_values.append(float(prompt_entropy.detach().item()))
                if self.prompt_entropy_weight != 0.0:
                    entropy_term = -self.prompt_entropy_weight * prompt_entropy
                    loss = loss + entropy_term
                    prompt_entropy_penalties.append(float(entropy_term.detach().item()))

            # 
            epoch_loss += loss.item()
            for task, task_loss in task_losses.items():
                epoch_task_losses[task] += task_loss.item()
                self.task_loss_history[task].append(task_loss.item())

            # 
            if self.use_contrastive and contrastive_features:
                contrastive_loss, contrastive_logs, contrastive_state = self._compute_contrastive_loss(
                    contrastive_features,
                    labels,
                    task_losses,
                    batch_idx,
                    input_ids,
                    attention_mask,
                    token_type_ids
                )
                if contrastive_loss is not None:
                    loss = loss + contrastive_loss
                    for task, value in contrastive_logs.items():
                        epoch_contrastive_losses[task] += value
                        contrastive_counts[task] += 1
                if contrastive_state:
                    for key, value in contrastive_state.items():
                        contrastive_state_sums[key] = contrastive_state_sums.get(key, 0.0) + value
                        contrastive_state_counts[key] = contrastive_state_counts.get(key, 0) + 1

            # 
            loss.backward()

            # 
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.global_step += 1
            num_batches += 1

            # 
            if self.global_step % self.config.log_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_training_step(self.global_step, loss.item(), current_lr)

            # Momentum 
            if self.use_contrastive and self.momentum_model is not None:
                self._momentum_update()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # 
        avg_loss = epoch_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in epoch_task_losses.items()}
        avg_contrastive_losses = {}
        if self.use_contrastive and self.contrastive_tasks:
            for task in self.contrastive_tasks:
                if contrastive_counts.get(task, 0) > 0:
                    avg_contrastive_losses[task] = epoch_contrastive_losses[task] / contrastive_counts[task]

        # epoch
        epoch_metrics = {
            'total_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        epoch_metrics.update({f'{task}_loss': loss for task, loss in avg_task_losses.items()})
        if avg_contrastive_losses:
            epoch_metrics.update({f'contrastive_loss_{task}': loss for task, loss in avg_contrastive_losses.items()})
            epoch_metrics['contrastive_total_loss'] = sum(avg_contrastive_losses.values())
        if contrastive_state_sums:
            for key, value in contrastive_state_sums.items():
                epoch_metrics[key] = value / contrastive_state_counts[key]
        if prompt_stat_sums:
            for key, value in prompt_stat_sums.items():
                epoch_metrics[key] = value / max(prompt_stat_counts.get(key, 1), 1)
        if prompt_entropy_values:
            epoch_metrics['prompt_entropy_mean'] = sum(prompt_entropy_values) / len(prompt_entropy_values)
        if prompt_entropy_penalties:
            epoch_metrics['prompt_entropy_penalty_mean'] = sum(prompt_entropy_penalties) / len(prompt_entropy_penalties)
        if self.current_prompt_alpha is not None:
            epoch_metrics['prompt_alpha'] = float(self.current_prompt_alpha)
        if getattr(self.model, 'use_prompt_mixing', False):
            self._maybe_adjust_prompt_dropout(epoch_metrics)

        return epoch_metrics

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """"""
        return self.evaluator.evaluate(self.model, val_loader, self.device, mode='validation')

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """."""
        num_training_steps = len(train_loader) * num_epochs
        self.setup_scheduler(num_training_steps)

        # 
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.logger.log_epoch_start(epoch, num_epochs)
            prompt_alpha = self._apply_prompt_alpha_schedule()
            if prompt_alpha is not None:
                self.logger.logger.info(
                    f"Prompt alpha updated to {prompt_alpha:.4f} (epoch {epoch + 1}/{num_epochs})"
                )
            # poch
            train_metrics = self.train_epoch(train_loader)

            # 
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            # epoch
            epoch_record = {'epoch': epoch, **train_metrics}
            self.training_history.append(epoch_record)
            self.logger.log_epoch_end(epoch, train_metrics)

            self.save_checkpoint(tag='last')

            # 
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(tag=f'checkpoint_epoch_{self.current_epoch+1}')

        # 
        self.logger.logger.info(f"\nTraining completed!")

    def save_checkpoint(self, tag: Optional[str] = 'last'):
        """"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'task_weights': self.dynamic_weights,
            'config': self.config.to_dict()
        }
        prompt_state = None
        if getattr(self.config, 'use_prompt_mixing', False):
            prompt_state = self.model.get_prompt_state()
            if prompt_state:
                checkpoint['prompt_state_dict'] = prompt_state

        if not tag:
            filename = f'checkpoint_epoch_{self.current_epoch+1}.pt'
        elif tag.endswith('.pt'):
            filename = tag
        elif tag == 'last':
            filename = 'last_model.pt'
        else:
            filename = f'{tag}.pt'

        checkpoint_path = os.path.join(self.config.saved_model_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        self.logger.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if prompt_state:
            prompt_suffix = filename.replace('.pt', '_prompt.pt')
            prompt_path = os.path.join(self.config.saved_model_dir, prompt_suffix)
            torch.save(prompt_state, prompt_path)
            self.logger.logger.info(f"Prompt adapter saved: {prompt_path}")

    def load_model_weights(self, checkpoint_path: str) -> Optional[Dict]:
        """Load model weights for evaluation or checkpoint reuse."""
        if not os.path.exists(checkpoint_path):
            self.logger.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if getattr(self.config, 'use_prompt_mixing', False):
            prompt_state = checkpoint.get('prompt_state_dict')
            if not prompt_state:
                prompt_file = checkpoint_path.replace('.pt', '_prompt.pt')
                if os.path.exists(prompt_file):
                    prompt_state = torch.load(prompt_file, map_location='cpu')
            if prompt_state:
                self.model.load_prompt_state(prompt_state)
        return checkpoint

    def load_checkpoint(self, checkpoint_path: str):
        """"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if getattr(self.config, 'use_prompt_mixing', False):
            prompt_state = checkpoint.get('prompt_state_dict')
            if not prompt_state:
                prompt_file = checkpoint_path.replace('.pt', '_prompt.pt')
                if os.path.exists(prompt_file):
                    prompt_state = torch.load(prompt_file, map_location='cpu')
            if prompt_state:
                self.model.load_prompt_state(prompt_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint.get('global_step', self.global_step)
        self.current_epoch = checkpoint.get('epoch', self.current_epoch)
        self.dynamic_weights = checkpoint.get('task_weights', self.config.task_weights)

        self.logger.logger.info(f"Loaded checkpoint from epoch {self.current_epoch+1}")

    def get_training_summary(self) -> Dict:
        """"""
        return {
            'total_epochs': self.current_epoch + 1,
            'global_steps': self.global_step,
            'final_task_weights': self.dynamic_weights,
            'training_history': self.training_history
        }




