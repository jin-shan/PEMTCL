# -*- coding: utf-8 -*-
"""
?"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel
import math
from typing import Dict


class PEMTCLModel(nn.Module):
    """."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task_weights = config.task_weights
        self.model_type = getattr(config, 'model_type', 'roberta')

        self.enable_contrastive = getattr(config, 'enable_contrastive', False)
        self.contrastive_projection_dim = getattr(config, 'contrastive_projection_dim', 256)
        self.contrastive_projection_hidden = getattr(config, 'contrastive_projection_hidden', 0)
        self.contrastive_dropout = getattr(config, 'contrastive_dropout', 0.1)
        self.contrastive_normalize = getattr(config, 'contrastive_normalize', True)
        self.contrastive_use_dual_view = getattr(config, 'contrastive_use_dual_view', True)
        self.contrastive_noise_scale = getattr(config, 'contrastive_noise_scale', 0.0)
        self.contrastive_view_dropout_layer = nn.Dropout(
            getattr(config, 'contrastive_view_dropout', self.contrastive_dropout)
        ) if self.enable_contrastive else None
        self.use_prompt_mixing = bool(getattr(config, 'use_prompt_mixing', False))
        self.prompt_length = max(1, int(getattr(config, 'prompt_length', 20)))
        self.num_source_prompts = max(1, int(getattr(config, 'num_source_prompts', 4)))
        self.prompt_attn_hidden = int(getattr(config, 'prompt_attn_hidden', 256))
        self.prompt_lr = float(getattr(config, 'prompt_lr', 5e-4))
        self.prompt_weight_decay = float(getattr(config, 'prompt_weight_decay', 0.01))
        default_alpha = 0.0 if self.use_prompt_mixing else 1.0
        self.prompt_alpha = float(getattr(config, 'prompt_alpha_init', default_alpha))
        self.prompt_alpha_max = float(getattr(config, 'prompt_alpha_max', self.prompt_alpha))
        self.prompt_attn_topk = int(getattr(config, 'prompt_attn_topk', 0))
        self.prompt_attn_topk_ratio = float(getattr(config, 'prompt_attn_topk_ratio', 0.5))
        self.prompt_attn_threshold = float(getattr(config, 'prompt_attn_threshold', 0.0))
        self.prompt_attn_dropout_prob = float(getattr(config, 'prompt_attn_dropout', 0.0))
        per_task_topk = getattr(config, 'prompt_attn_topk_per_task', {})
        if not isinstance(per_task_topk, dict):
            per_task_topk = {}
        self.prompt_attn_topk_per_task = per_task_topk
        self.source_prompts = None
        self.shared_target_prompt = None
        self.target_prompts = None
        self.prompt_attn = None
        self.task_prompt_embeddings = None

        if self.model_type == 'roberta':
            backbone_path = config.roberta_model_path
            self.backbone = AutoModel.from_pretrained(backbone_path)
        elif self.model_type == 'macbert':
            backbone_path = getattr(config, 'macbert_model_path', config.bert_model_path)
            self.backbone = BertModel.from_pretrained(backbone_path)
        elif self.model_type == 'bert':
            self.backbone = BertModel.from_pretrained(config.bert_model_path)
        else:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. Expected one of ['bert', 'roberta', 'macbert']."
            )

        hidden_size = self.backbone.config.hidden_size
        if self.use_prompt_mixing:
            self._init_prompt_modules(hidden_size)
        self.hidden_size = hidden_size

        self.task_classifiers = nn.ModuleDict(
            {
                "toxic": nn.Sequential(
                    nn.Dropout(config.classifier_dropout),
                    nn.Linear(hidden_size, 2),
                ),
                "toxic_type": nn.Sequential(
                    nn.Dropout(config.classifier_dropout),
                    nn.Linear(hidden_size, 2),
                ),
                "expression": nn.Sequential(
                    nn.Dropout(config.classifier_dropout),
                    nn.Linear(hidden_size, 3),
                ),
                "target": nn.Sequential(
                    nn.Dropout(config.classifier_dropout),
                    nn.Linear(hidden_size, 5),
                ),
            }
        )

        #  Target 
        if getattr(config, 'target_pos_weight', None):
            self.register_buffer(
                'target_pos_weight_tensor',
                torch.tensor(config.target_pos_weight, dtype=torch.float)
            )
        else:
            self.target_pos_weight_tensor = None

        # 
        if config.task_uncertainty:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.task_classifiers.keys()
            })
        else:
            self.log_vars = None

        self.contrastive_tasks = []
        if self.enable_contrastive:
            # ?0 ?            task_weight_cfg = getattr(config, 'contrastive_task_weights', {})
            self.contrastive_tasks = [
                task for task in self.task_classifiers.keys()
                if task_weight_cfg.get(task, 0.0) > 0.0 and self.task_weights.get(task, 0.0) > 0.0
            ]
            self.task_projections = nn.ModuleDict()
            proj_hidden = max(0, int(self.contrastive_projection_hidden))
            proj_dim = max(1, int(self.contrastive_projection_dim))
            for task in self.contrastive_tasks:
                layers = []
                if proj_hidden > 0:
                    layers.append(nn.Linear(hidden_size, proj_hidden))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(self.contrastive_dropout))
                    layers.append(nn.Linear(proj_hidden, proj_dim))
                else:
                    layers.append(nn.Linear(hidden_size, proj_dim))
                self.task_projections[task] = nn.Sequential(*layers)
        else:
            self.task_projections = None

        self._init_weights()

    def _init_prompt_modules(self, hidden_size: int):
        attn_hidden = max(1, int(getattr(self.config, 'prompt_attn_hidden', hidden_size)))
        self.source_prompts = nn.Parameter(
            torch.randn(self.num_source_prompts, self.prompt_length, hidden_size) * 0.02
        )
        self.shared_target_prompt = nn.Parameter(
            torch.randn(self.prompt_length, hidden_size) * 0.02
        )
        task_names = getattr(self.config, 'task_list', list(self.task_weights.keys()))
        self.target_prompts = nn.ParameterDict({
            task: nn.Parameter(torch.randn(self.prompt_length, hidden_size) * 0.02)
            for task in task_names
        })
        self.prompt_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1)
        )
        self.task_prompt_embeddings = nn.ParameterDict({
            task: nn.Parameter(torch.randn(hidden_size) * 0.02)
            for task in task_names
        })

    def _set_module_grad(self, module, requires_grad: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _set_parameter_grad(self, param, requires_grad: bool):
        if param is None:
            return
        param.requires_grad = requires_grad

    def apply_training_stage(self, stage: str):
        """Apply the fixed PEMTCL stage-specific training policy."""
        if not getattr(self, 'use_prompt_mixing', False):
            return
        stage = (stage or 'stage_b').lower()
        if stage == 'stage_a':
            self._set_module_grad(self.backbone, False)
            self._set_module_grad(self.task_classifiers, False)
            if hasattr(self, 'task_projections') and self.task_projections is not None:
                self._set_module_grad(self.task_projections, False)
            self._set_parameter_grad(self.source_prompts, True)
            self._set_parameter_grad(self.shared_target_prompt, False)
            if self.target_prompts is not None:
                for param in self.target_prompts.values():
                    self._set_parameter_grad(param, False)
            if self.task_prompt_embeddings is not None:
                for param in self.task_prompt_embeddings.values():
                    self._set_parameter_grad(param, False)
            self._set_module_grad(self.prompt_attn, True)
        elif stage == 'stage_b':
            self._set_module_grad(self.backbone, True)
            self._set_module_grad(self.task_classifiers, True)
            if hasattr(self, 'task_projections') and self.task_projections is not None:
                self._set_module_grad(self.task_projections, True)
            self._set_parameter_grad(self.source_prompts, False)
            self._set_parameter_grad(self.shared_target_prompt, True)
            if self.target_prompts is not None:
                for param in self.target_prompts.values():
                    self._set_parameter_grad(param, True)
            if self.task_prompt_embeddings is not None:
                for param in self.task_prompt_embeddings.values():
                    self._set_parameter_grad(param, True)
            self._set_module_grad(self.prompt_attn, True)
        else:
            raise ValueError(f"Unknown PEMTCL stage: {stage}")

    def _build_instance_prompt_shared(self, pooled_input: torch.Tensor):
        """
        ?        """
        if (
            not self.use_prompt_mixing
            or self.source_prompts is None
            or self.shared_target_prompt is None
            or self.prompt_attn is None
        ):
            return None, None

        if self.target_prompts is not None and len(self.target_prompts) > 0:
            target_stack = torch.stack([param for param in self.target_prompts.values()], dim=0)
            target_prompt = target_stack.mean(dim=0)
        else:
            target_prompt = self.shared_target_prompt

        B, H = pooled_input.size()
        source = self.source_prompts  # (T, Lp, H)
        source_pooled = source.mean(dim=1)  # (T, H)
        target_pooled = target_prompt.mean(dim=0, keepdim=True)  # (1, H)
        all_pooled = torch.cat([source_pooled, target_pooled], dim=0)  # (T+1, H)

        prompts_expand = all_pooled.unsqueeze(0).expand(B, -1, -1)  # (B, T+1, H)
        input_expand = pooled_input.unsqueeze(1).expand_as(prompts_expand)  # (B, T+1, H)
        concat = torch.cat([input_expand, prompts_expand], dim=-1)  # (B, T+1, 2H)

        scores = self.prompt_attn(concat).squeeze(-1)  # (B, T+1)
        attn_weight = torch.softmax(scores, dim=-1)  # (B, T+1)
        attn_weight = self._apply_prompt_topk(attn_weight)

        all_prompts_full = torch.cat([source, target_prompt.unsqueeze(0)], dim=0)  # (T+1, Lp, H)
        weighted = attn_weight.view(B, -1, 1, 1) * all_prompts_full.unsqueeze(0)
        inst_prompt = weighted.sum(dim=1)  # (B, Lp, H)
        return inst_prompt, attn_weight

    def _build_instance_prompt_for_task(self, task_name: str, pooled_input: torch.Tensor):
        """
         gating?        """
        if (
            not self.use_prompt_mixing
            or self.source_prompts is None
            or self.shared_target_prompt is None
            or self.prompt_attn is None
        ):
            return None, None

        B, H = pooled_input.size()
        source = self.source_prompts
        if self.target_prompts is not None and task_name in self.target_prompts:
            target_prompt = self.target_prompts[task_name]
        else:
            target_prompt = self.shared_target_prompt
        task_input = pooled_input
        if self.task_prompt_embeddings is not None and task_name in self.task_prompt_embeddings:
            task_input = task_input + self.task_prompt_embeddings[task_name].view(1, H)

        source_pooled = source.mean(dim=1)
        target_pooled = target_prompt.mean(dim=0, keepdim=True)
        all_pooled = torch.cat([source_pooled, target_pooled], dim=0)

        prompts_expand = all_pooled.unsqueeze(0).expand(B, -1, -1)
        input_expand = task_input.unsqueeze(1).expand_as(prompts_expand)
        concat = torch.cat([input_expand, prompts_expand], dim=-1)

        scores = self.prompt_attn(concat).squeeze(-1)
        attn_weight = torch.softmax(scores, dim=-1)
        attn_weight = self._apply_prompt_topk(attn_weight, self._get_task_topk(task_name))

        all_prompts_full = torch.cat([source, target_prompt.unsqueeze(0)], dim=0)
        weighted = attn_weight.view(B, -1, 1, 1) * all_prompts_full.unsqueeze(0)
        inst_prompt = weighted.sum(dim=1)
        return inst_prompt, attn_weight

    def _summarize_prompt_stats(self, attn_weight: torch.Tensor) -> Dict[str, float]:
        stats = {}
        entropy = -(attn_weight * torch.log(attn_weight + 1e-8)).sum(dim=-1).mean()
        stats['entropy'] = float(entropy.detach().cpu().item())
        mean_weights = attn_weight.mean(dim=0).detach().cpu()
        num_sources = mean_weights.size(0) - 1
        for idx in range(num_sources):
            stats[f'source_{idx}_weight'] = float(mean_weights[idx].item())
        stats['target_weight'] = float(mean_weights[-1].item())
        stats['max_weight'] = float(attn_weight.max(dim=-1).values.mean().detach().cpu().item())
        active = (attn_weight > 0).float().sum(dim=-1).mean()
        stats['active_prompts'] = float(active.detach().cpu().item())
        return stats

    def _get_task_topk(self, task_name: str) -> int:
        if not self.prompt_attn_topk_per_task:
            return int(getattr(self, 'prompt_attn_topk', 0))
        return int(self.prompt_attn_topk_per_task.get(task_name, getattr(self, 'prompt_attn_topk', 0)))

    def _apply_prompt_topk(self, attn_weight: torch.Tensor, topk: int = None) -> torch.Tensor:
        total = attn_weight.size(-1)
        threshold = max(0.0, float(getattr(self, 'prompt_attn_threshold', 0.0)))
        ratio = max(0.0, float(getattr(self, 'prompt_attn_topk_ratio', 0.0)))
        dropout_prob = max(0.0, float(getattr(self, 'prompt_attn_dropout_prob', 0.0)))

        if self.training and dropout_prob > 0.0:
            keep_mask = (torch.rand_like(attn_weight) > dropout_prob).float()
            attn_weight = attn_weight * keep_mask + 1e-8

        if threshold > 0.0:
            thresh_mask = (attn_weight >= threshold).float()
            attn_weight = attn_weight * thresh_mask + 1e-8

        if topk is None or topk <= 0:
            k = int(getattr(self, 'prompt_attn_topk', 0))
        else:
            k = int(topk)
        if k <= 0 and ratio > 0.0:
            k = max(1, int(math.ceil(total * min(ratio, 1.0))))
        if k <= 0 or k >= total:
            norm = attn_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return attn_weight / norm

        topk_values, topk_indices = torch.topk(attn_weight, k, dim=-1)
        mask = torch.zeros_like(attn_weight)
        mask.scatter_(-1, topk_indices, 1.0)
        pruned = attn_weight * mask
        norm = pruned.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return pruned / norm

    def _get_transformer_pooled(self, outputs):
        """transformerlast_hidden_stateooled"""
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
            pooled_output = getattr(outputs, 'pooler_output', None)
            if pooled_output is None:
                pooled_output = last_hidden_state[:, 0, :]
        elif isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
            pooled_output = None
            if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                pooled_output = outputs[1]
            if pooled_output is None:
                pooled_output = last_hidden_state[:, 0, :]
        else:
            last_hidden_state = outputs
            pooled_output = last_hidden_state[:, 0, :]
        return last_hidden_state, pooled_output

    def iter_prompt_parameters(self):
        """Return trainable prompt parameters."""
        if not self.use_prompt_mixing:
            return []
        params = []
        if self.source_prompts is not None and self.source_prompts.requires_grad:
            params.append(('source_prompts', self.source_prompts))
        if self.shared_target_prompt is not None and self.shared_target_prompt.requires_grad:
            params.append(('shared_target_prompt', self.shared_target_prompt))
        if self.target_prompts is not None:
            for task, param in self.target_prompts.items():
                if param.requires_grad:
                    params.append((f'target_prompts.{task}', param))
        if self.prompt_attn is not None:
            for name, param in self.prompt_attn.named_parameters():
                if param.requires_grad:
                    params.append((f'prompt_attn.{name}', param))
        if self.task_prompt_embeddings is not None:
            for task, param in self.task_prompt_embeddings.items():
                if param.requires_grad:
                    params.append((f'task_prompt_embeddings.{task}', param))
        return params

    def set_prompt_alpha(self, alpha: float):
        """ prompt """
        if not self.use_prompt_mixing:
            self.prompt_alpha = 1.0
            return
        max_alpha = float(getattr(self.config, 'prompt_alpha_max', 1.0))
        clipped = max(0.0, min(float(alpha), max_alpha))
        self.prompt_alpha = clipped

    def set_prompt_dropout(self, dropout: float):
        """ prompt ?dropout"""
        if not self.use_prompt_mixing:
            return
        self.prompt_attn_dropout_prob = max(0.0, min(0.9, float(dropout)))

    def get_prompt_alpha(self) -> float:
        """ prompt """
        return float(getattr(self, 'prompt_alpha', 1.0))

    def get_prompt_state(self):
        """Export prompt parameters."""
        if not self.use_prompt_mixing:
            return None
        state = {}
        if self.source_prompts is not None:
            state['source_prompts'] = self.source_prompts.detach().cpu()
        if self.shared_target_prompt is not None:
            state['shared_target_prompt'] = self.shared_target_prompt.detach().cpu()
        if self.target_prompts is not None:
            state['target_prompts'] = {
                task: param.detach().cpu()
                for task, param in self.target_prompts.items()
            }
        if self.prompt_attn is not None:
            state['prompt_attn'] = self.prompt_attn.state_dict()
        if self.task_prompt_embeddings is not None:
            state['task_prompt_embeddings'] = {
                task: param.detach().cpu()
                for task, param in self.task_prompt_embeddings.items()
            }
        return state

    def load_prompt_state(self, state: dict):
        """Load prompt parameters."""
        if not state or not self.use_prompt_mixing:
            return
        device = self.source_prompts.device if self.source_prompts is not None else self.config.device
        if 'source_prompts' in state and self.source_prompts is not None:
            self.source_prompts.data.copy_(state['source_prompts'].to(device))
        if 'shared_target_prompt' in state and self.shared_target_prompt is not None:
            self.shared_target_prompt.data.copy_(state['shared_target_prompt'].to(device))
        if 'target_prompts' in state and self.target_prompts is not None:
            for task, value in state['target_prompts'].items():
                if task in self.target_prompts:
                    self.target_prompts[task].data.copy_(value.to(device))
        if 'prompt_attn' in state and self.prompt_attn is not None:
            self.prompt_attn.load_state_dict(state['prompt_attn'])
        if 'task_prompt_embeddings' in state and self.task_prompt_embeddings is not None:
            for task, value in state['task_prompt_embeddings'].items():
                if task in self.task_prompt_embeddings:
                    self.task_prompt_embeddings[task].data.copy_(value.to(device))

    def _init_weights(self):
        """"""
        for classifier in self.task_classifiers.values():
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.enable_contrastive and self.task_projections is not None:
            for projection in self.task_projections.values():
                for module in projection.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)

    def _build_contrastive_view(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """1.1"""
        if not self.enable_contrastive:
            return pooled_output
        augmented = pooled_output.clone()

        if self.contrastive_view_dropout_layer is not None and self.training:
            augmented = self.contrastive_view_dropout_layer(augmented)
            augmented = self.contrastive_view_dropout_layer(augmented)

        if self.training and getattr(self, 'contrastive_noise_scale', 0.0) > 0:
            noise = torch.randn_like(augmented) * self.contrastive_noise_scale
            augmented = augmented + noise
        return augmented


    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """"""
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)

        prompt_stats = None
        prompt_entropy_value = None
        per_task_prompt_vectors = {}
        per_task_attn_weights = {}
        prompt_alpha_value = float(getattr(self, 'prompt_alpha', 1.0))
        transformer_hidden_states = None
        transformer_attentions = None

        use_prompt = (
            self.use_prompt_mixing
            and hasattr(self.backbone, 'embeddings')
            and self.source_prompts is not None
            and self.shared_target_prompt is not None
        )
        if use_prompt:
            embeddings = self.backbone.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            pooled_input = embeddings.mean(dim=1)
            inst_prompt, attn_weight = self._build_instance_prompt_shared(pooled_input)
            prompt_alpha_value = float(getattr(self, 'prompt_alpha', 1.0))
            for task_name in self.task_classifiers.keys():
                inst_task, attn_task = self._build_instance_prompt_for_task(task_name, pooled_input)
                if inst_task is not None:
                    per_task_prompt_vectors[task_name] = inst_task.mean(dim=1)
                if attn_task is not None:
                    per_task_attn_weights[task_name] = attn_task
            if inst_prompt is not None:
                if prompt_alpha_value != 1.0:
                    inst_prompt = inst_prompt * prompt_alpha_value
                prompt_mask = torch.ones(
                    inst_prompt.size(0),
                    inst_prompt.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                inputs_embeds = torch.cat([inst_prompt, embeddings], dim=1)
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            else:
                inputs_embeds = embeddings
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=None
            )
            if attn_weight is not None:
                shared_entropy = -(attn_weight * torch.log(attn_weight + 1e-8)).sum(dim=-1).mean()
                stats = self._summarize_prompt_stats(attn_weight.detach())
                stats['alpha'] = float(prompt_alpha_value)
                if per_task_attn_weights:
                    prompt_stats = {}
                    entropies = []
                    for task, task_attn in per_task_attn_weights.items():
                        per_stats = self._summarize_prompt_stats(task_attn.detach())
                        per_stats['alpha'] = float(prompt_alpha_value)
                        per_stats['topk'] = self._get_task_topk(task)
                        if self.task_prompt_embeddings is not None and task in self.task_prompt_embeddings:
                            per_stats['task_embedding_norm'] = float(self.task_prompt_embeddings[task].norm().item())
                        prompt_stats[task] = per_stats
                        entropies.append(
                            -(task_attn * torch.log(task_attn + 1e-8)).sum(dim=-1).mean()
                        )
                    if entropies:
                        prompt_entropy_value = torch.stack(entropies).mean()
                    else:
                        prompt_entropy_value = shared_entropy
                else:
                    prompt_stats = {task: stats for task in self.task_classifiers.keys()}
                    prompt_entropy_value = shared_entropy
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        if hasattr(outputs, 'last_hidden_state'):
            pooled_output = getattr(outputs, 'pooler_output', None)
            if pooled_output is None:
                pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            _, pooled_output = self._get_transformer_pooled(outputs)
        transformer_hidden_states = getattr(outputs, 'hidden_states', None)
        transformer_attentions = getattr(outputs, 'attentions', None)

        task_features = {}
        prompt_alpha_value = float(getattr(self, 'prompt_alpha', 1.0))
        for task_name in self.task_classifiers.keys():
            addition = per_task_prompt_vectors.get(task_name)
            if addition is not None:
                task_features[task_name] = {'pooled': pooled_output + prompt_alpha_value * addition}
            else:
                task_features[task_name] = {'pooled': pooled_output}
        contrastive_views = {}
        if self.enable_contrastive and self.task_projections is not None and self.contrastive_use_dual_view:
            for task_name, feats in task_features.items():
                contrastive_views[task_name] = self._build_contrastive_view(feats['pooled'])

        # 
        task_outputs = {}
        task_losses = {}
        contrastive_outputs = {}

        for task_name, classifier in self.task_classifiers.items():
            pooled_output = task_features[task_name]['pooled']
            logits = classifier(pooled_output)
            task_outputs[task_name] = logits

            if labels is not None and task_name in labels:
                if self.task_weights.get(task_name, 0) <= 0:
                    continue
                if task_name == 'target':
                    # TargetBCEWithLogitsLoss?                    target_labels = labels[task_name].float()
                    if self.target_pos_weight_tensor is not None:
                        pos_weight = self.target_pos_weight_tensor.to(logits.device)
                        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    else:
                        loss_fn = nn.BCEWithLogitsLoss()
                    task_loss = loss_fn(logits, target_labels)
                elif task_name in ('toxic_type', 'expression'):
                    # ?BCEWithLogitsLoss  one-hot 0?                    task_labels = labels[task_name].float()
                    loss_fn = nn.BCEWithLogitsLoss()
                    task_loss = loss_fn(logits, task_labels)
                else:
                    # CrossEntropyLoss
                    task_labels = labels[task_name]
                    weight_tensor = None
                    if task_name == 'expression' and getattr(self.config, 'expression_class_weight', None):
                        weight_tensor = torch.tensor(
                            self.config.expression_class_weight,
                            dtype=torch.float,
                            device=logits.device
                        )
                    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
                    task_loss = loss_fn(logits, task_labels)

                if self.log_vars is not None:
                    precision = torch.exp(-self.log_vars[task_name])
                    task_loss = precision * task_loss + self.log_vars[task_name] / 2

                task_losses[task_name] = task_loss

            if self.enable_contrastive and self.task_projections is not None:
                if task_name in self.task_projections:
                    primary_proj = self.task_projections[task_name](pooled_output)
                    positive_proj = None
                    if task_name in contrastive_views:
                        positive_proj = self.task_projections[task_name](contrastive_views[task_name])
                    if self.contrastive_normalize:
                        primary_proj = F.normalize(primary_proj, dim=-1)
                        if positive_proj is not None:
                            positive_proj = F.normalize(positive_proj, dim=-1)
                    contrastive_outputs[task_name] = {
                        'anchor': primary_proj,
                        'positive': positive_proj
                    }

        # ?        total_loss = 0
        if task_losses:
            if self.log_vars is None:
                for task_name, loss in task_losses.items():
                    total_loss += self.task_weights[task_name] * loss
            else:
                total_loss = sum(task_losses.values())

        return {
            'task_outputs': task_outputs,
            'task_losses': task_losses,
            'total_loss': total_loss,
            'pooled_output': pooled_output,
            'contrastive_features': contrastive_outputs if contrastive_outputs else None,
            'hidden_states': transformer_hidden_states,
            'attentions': transformer_attentions,
            'prompt_stats': prompt_stats,
            'prompt_entropy': prompt_entropy_value
        }

    def get_task_predictions(self, outputs, apply_softmax=True):
        """"""
        predictions = {}
        task_outputs = outputs['task_outputs']

        for task_name, logits in task_outputs.items():
            if task_name in ('target', 'toxic_type', 'expression'):
                # ?/ one-hot0?sigmoid 
                predictions[task_name] = torch.sigmoid(logits)
            else:
                # oftmax
                if apply_softmax:
                    predictions[task_name] = torch.softmax(logits, dim=-1)
                else:
                    predictions[task_name] = logits

        return predictions

    def get_task_classes(self, task_name):
        """"""
        if task_name == 'toxic':
            return 2
        elif task_name == 'toxic_type':
            return 2
        elif task_name == 'expression':
            return 3
        elif task_name == 'target':
            return 5
        else:
            raise ValueError(f"Unknown task: {task_name}")





