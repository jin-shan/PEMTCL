# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import multilabel_confusion_matrix
import json
from typing import Dict, List, Tuple

class PEMTCLEvaluator:
    """"""

    def __init__(self, config):
        self.config = config
        configured_tasks = list(getattr(config, 'task_list', ['toxic', 'toxic_type', 'expression', 'target']))
        weights = getattr(config, 'task_weights', {})
        self.task_names = [t for t in configured_tasks if weights.get(t, 0) > 0]
        self.task_metrics = {}
        #  ToxiCN_ex / toxic-detection-mainoxic_type/expression  one-hot0
        self.vector_tasks = {'target', 'toxic_type', 'expression'}

    def _decode_abstain_onehot(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        (N, C) -> (N, C) one-hot  0bstain?        ax_prob < threshold => ? argmax one-hot?        """
        if probs.size == 0:
            return probs.astype(int)
        max_probs = probs.max(axis=1)
        argmax = probs.argmax(axis=1)
        out = np.zeros_like(probs, dtype=int)
        active = max_probs >= float(threshold)
        if np.any(active):
            idx = np.nonzero(active)[0]
            out[idx, argmax[idx]] = 1
        return out

    def _onehot_to_id(self, onehot: np.ndarray) -> np.ndarray:
        """
        ?one-hot / ? ?id0 -> 0?argmax+1?        ??        """
        if onehot.size == 0:
            return np.array([], dtype=int)
        if onehot.ndim == 1:
            onehot = onehot.reshape(1, -1)
        sums = onehot.sum(axis=1)
        ids = np.zeros(onehot.shape[0], dtype=int)
        active = sums > 0
        if np.any(active):
            ids[active] = onehot[active].argmax(axis=1) + 1
        return ids

    def evaluate(self, model, data_loader, device, mode='test'):
        """"""
        model.eval()
        all_predictions = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}
        total_loss = 0
        num_batches = 0

        analysis_dir = getattr(self.config, 'analysis_dir', None)
        analysis_enabled = bool(analysis_dir)
        if analysis_enabled:
            analysis_path = Path(analysis_dir)
            analysis_path.mkdir(parents=True, exist_ok=True)
            misclassified_samples = []
            batch_index_offset = 0
        else:
            analysis_path = None

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}

                # 
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                total_loss += outputs['total_loss'].item()
                num_batches += 1

                # 
                predictions = model.get_task_predictions(outputs, apply_softmax=False)
                batch_pred_map = {}
                batch_label_map = {}

                # 
                for task_name in self.task_names:
                    if task_name == 'target':
                        # ?                        preds = (predictions[task_name] > 0.5).cpu().numpy().astype(int)
                        labels_np = labels[task_name].cpu().numpy().astype(int)
                    elif task_name in ('toxic_type', 'expression'):
                        probs = predictions[task_name].cpu().numpy()
                        preds = self._decode_abstain_onehot(probs, threshold=0.5)
                        labels_np = labels[task_name].cpu().numpy().astype(int)
                    else:
                        # ?                        preds = torch.argmax(predictions[task_name], dim=-1).cpu().numpy()
                        labels_np = labels[task_name].cpu().numpy()

                    all_predictions[task_name].extend(preds)
                    all_labels[task_name].extend(labels_np)
                    if analysis_enabled:
                        batch_pred_map[task_name] = preds
                        batch_label_map[task_name] = labels_np

                if analysis_enabled:
                    texts = batch.get('texts', [])
                    original_info = batch.get('original_info', {})
                    batch_size = len(texts)
                    for idx in range(batch_size):
                        record = {
                            'index': batch_index_offset + idx,
                            'text': texts[idx],
                            'true': {},
                            'pred': {},
                            'mismatches': []
                        }
                        for task_name in self.task_names:
                            if task_name in self.vector_tasks:
                                true_vec = batch_label_map[task_name][idx].tolist()
                                pred_vec = batch_pred_map[task_name][idx].tolist()
                                record['true'][task_name] = true_vec
                                record['pred'][task_name] = pred_vec
                                if true_vec != pred_vec:
                                    record['mismatches'].append(task_name)
                            else:
                                true_val = int(batch_label_map[task_name][idx])
                                pred_val = int(batch_pred_map[task_name][idx])
                                record['true'][task_name] = true_val
                                record['pred'][task_name] = pred_val
                                if true_val != pred_val:
                                    record['mismatches'].append(task_name)
                        if original_info:
                            record['original_info'] = {k: original_info.get(k, [])[idx] for k in original_info}
                        if record['mismatches']:
                            misclassified_samples.append(record)
                    batch_index_offset += batch_size

        # 
        metrics = self._compute_metrics(all_predictions, all_labels)
        metrics['total_loss'] = total_loss / num_batches

        if analysis_enabled:
            self._export_analysis(analysis_path, all_labels, all_predictions, misclassified_samples)

        # 
        self._print_results(metrics, mode)

        return metrics

    def _compute_metrics(self, predictions: Dict[str, List], labels: Dict[str, List]) -> Dict[str, float]:
        """"""
        metrics = {}

        for task_name in self.task_names:
            y_true = np.array(labels[task_name])
            y_pred = np.array(predictions[task_name])

            if task_name in self.vector_tasks:
                # ?/ one-hot 
                metrics.update(self._compute_multilabel_metrics(task_name, y_true, y_pred))
            else:
                metrics.update(self._compute_classification_metrics(task_name, y_true, y_pred))

        # 
        active_tasks = []
        for task in self.task_names:
            weight = getattr(self.config, 'task_weights', {}).get(task, 1.0)
            if weight > 0 and f'{task}_f1_macro' in metrics:
                active_tasks.append(task)

        if not active_tasks:
            active_tasks = list(self.task_names)

        metrics['macro_f1'] = np.mean([metrics[f'{t}_f1_macro'] for t in active_tasks])
        metrics['macro_precision'] = np.mean([metrics[f'{t}_precision_macro'] for t in active_tasks])
        metrics['macro_recall'] = np.mean([metrics[f'{t}_recall_macro'] for t in active_tasks])

        return metrics

    def _compute_classification_metrics(self, task_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """"""
        metrics = {}

        #  0 
        metrics[f'{task_name}_accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{task_name}_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{task_name}_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{task_name}_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{task_name}_precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f'{task_name}_recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f'{task_name}_f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1 in enumerate(class_f1):
            metrics[f'{task_name}_class_{i}_f1'] = float(f1)

        return metrics

    def _compute_multilabel_metrics(self, task_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """."""
        metrics = {}

        # ?        metrics[f'{task_name}_accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{task_name}_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{task_name}_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{task_name}_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ?        metrics[f'{task_name}_precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics[f'{task_name}_recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics[f'{task_name}_f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        metrics[f'{task_name}_precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f'{task_name}_recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f'{task_name}_f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # F1
        label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1 in enumerate(label_f1):
            metrics[f'{task_name}_label_{i}_f1'] = f1

        return metrics

    def _print_results(self, metrics: Dict[str, float], mode: str):
        """"""
        print(f"\n{'='*80}")
        print(f"{mode.upper()} RESULTS")
        print(f"{'='*80}")

        for task_name in self.task_names:
            print(f"\n{task_name.upper()}:")
            print(f"  Accuracy: {metrics.get(f'{task_name}_accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get(f'{task_name}_precision', 0):.4f}")
            print(f"  Recall: {metrics.get(f'{task_name}_recall', 0):.4f}")
            print(f"  F1: {metrics.get(f'{task_name}_f1', 0):.4f}")

            # /one-hoticro/macro
            if task_name in self.vector_tasks:
                print(f"  F1 (micro): {metrics.get(f'{task_name}_f1_micro', 0):.4f}")
                print(f"  F1 (macro): {metrics.get(f'{task_name}_f1_macro', 0):.4f}")
                print(f"  Precision (macro): {metrics.get(f'{task_name}_precision_macro', 0):.4f}")
                print(f"  Recall (macro): {metrics.get(f'{task_name}_recall_macro', 0):.4f}")
            else:
                print(f"  Precision (macro): {metrics.get(f'{task_name}_precision_macro', 0):.4f}")
                print(f"  Recall (macro): {metrics.get(f'{task_name}_recall_macro', 0):.4f}")
                print(f"  F1 (macro): {metrics.get(f'{task_name}_f1_macro', 0):.4f}")

        # 
        print(f"\nOVERALL AVERAGE:")
        print(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"  Macro Precision: {metrics.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall: {metrics.get('macro_recall', 0):.4f}")
        print(f"  Total Loss: {metrics.get('total_loss', 0):.4f}")
        print(f"{'='*80}")

    def get_detailed_report(self, predictions: Dict[str, List], labels: Dict[str, List]) -> str:
        """."""
        report = []
        report.append("DETAILED CLASSIFICATION REPORT\n")
        report.append("="*80)

        for task_name in self.task_names:
            if task_name in self.vector_tasks:
                continue  #  one-hot / ?
            y_true = np.array(labels[task_name])
            y_pred = np.array(predictions[task_name])

            report.append(f"\n{task_name.upper()}:")
            report.append("-"*40)
            class_report = classification_report(
                y_true, y_pred,
                target_names=self._get_class_names(task_name),
                digits=4
            )
            report.append(class_report)

            # 
            cm = confusion_matrix(y_true, y_pred)
            report.append(f"\nConfusion Matrix:")
            report.append(str(cm))

        return "\n".join(report)

    def _get_class_names(self, task_name: str) -> List[str]:
        """"""
        if task_name == 'toxic':
            return ['Non-toxic', 'Toxic']
        elif task_name == 'toxic_type':
            return ['None'] + list(getattr(self.config, 'toxic_type_labels', []))
        elif task_name == 'expression':
            return ['None'] + list(getattr(self.config, 'expression_labels', []))
        else:
            return [f'Class_{i}' for i in range(5)]

    def save_results(self, metrics: Dict[str, float], file_path: str):
        """"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def compare_with_baseline(self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """."""
        comparison = {}

        for task_name in self.task_names:
            current_f1 = current_metrics.get(f'{task_name}_f1', 0)
            baseline_f1 = baseline_metrics.get(f'{task_name}_f1', 0)

            comparison[f'{task_name}_f1_diff'] = current_f1 - baseline_f1
            comparison[f'{task_name}_f1_improvement'] = (current_f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0

        # 
        current_macro_f1 = current_metrics.get('macro_f1', 0)
        baseline_macro_f1 = baseline_metrics.get('macro_f1', 0)
        comparison['macro_f1_diff'] = current_macro_f1 - baseline_macro_f1
        comparison['macro_f1_improvement'] = (current_macro_f1 - baseline_macro_f1) / baseline_macro_f1 * 100 if baseline_macro_f1 > 0 else 0

        return comparison

    def print_comparison(self, comparison: Dict[str, float]):
        """"""
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)

        for task_name in self.task_names:
            diff = comparison.get(f'{task_name}_f1_diff', 0)
            improvement = comparison.get(f'{task_name}_f1_improvement', 0)

            if diff > 0:
                print(f"{task_name.upper()}: +{diff:.4f} F1 (+{improvement:.2f}%)")
            else:
                print(f"{task_name.upper()}: {diff:.4f} F1 ({improvement:.2f}%)")

        macro_diff = comparison.get('macro_f1_diff', 0)
        macro_improvement = comparison.get('macro_f1_improvement', 0)
        print(f"\nMACRO F1: {macro_diff:+.4f} ({macro_improvement:+.2f}%)")
        print("="*80)

    def _export_analysis(self, analysis_path: Path, labels_map: Dict[str, List], preds_map: Dict[str, List], misclassified_samples: List[Dict]):
        """."""
        confusion_dir = analysis_path / "confusion_matrices"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        self._save_confusion_matrices(confusion_dir, labels_map, preds_map)

        error_dir = analysis_path / "error_cases"
        error_dir.mkdir(parents=True, exist_ok=True)
        self._save_error_samples(error_dir, misclassified_samples)

        if 'target' in self.task_names:
            self._save_target_label_stats(analysis_path, labels_map['target'], preds_map['target'])

    def _save_confusion_matrices(self, output_dir: Path, labels_map: Dict[str, List], preds_map: Dict[str, List]):
        """."""
        for task_name in ['toxic', 'toxic_type', 'expression']:
            if task_name not in self.task_names:
                continue

            y_true = np.array(labels_map[task_name])
            y_pred = np.array(preds_map[task_name])
            if y_true.size == 0:
                continue

            if task_name in ('toxic_type', 'expression'):
                y_true = self._onehot_to_id(y_true)
                y_pred = self._onehot_to_id(y_pred)

            labels = sorted({*y_true.tolist(), *y_pred.tolist()})
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            cm_file = output_dir / f"{task_name}_confusion_matrix.json"
            with open(cm_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'labels': labels,
                    'matrix': cm.astype(int).tolist()
                }, f, ensure_ascii=False, indent=2)

            try:
                import matplotlib.pyplot as plt  # type: ignore
                import seaborn as sns  # type: ignore

                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self._get_task_label_names(task_name, labels),
                            yticklabels=self._get_task_label_names(task_name, labels))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {task_name}')
                plt.tight_layout()
                plt.savefig(output_dir / f"{task_name}_confusion_matrix.png", dpi=200)
                plt.close()
            except Exception:
                pass

    def _get_task_label_names(self, task_name: str, indices: List[int]) -> List[str]:
        """"""
        mapping = {
            'toxic': getattr(self.config, 'toxic_labels', []),
            'toxic_type': getattr(self.config, 'toxic_type_labels', []),
            'expression': getattr(self.config, 'expression_labels', [])
        }
        labels_list = mapping.get(task_name, [])
        if task_name in ('toxic_type', 'expression'):
            labels_list = ['None'] + list(labels_list)
        return [labels_list[i] if i < len(labels_list) else str(i) for i in indices]

    def _save_error_samples(self, output_dir: Path, samples: List[Dict]):
        """."""
        buckets = {
            'toxic_false_negatives': [],
            'toxic_false_positives': [],
            'expression_confusions': {},
            'target_mismatches': []
        }

        expression_labels = ['None'] + list(getattr(self.config, 'expression_labels', []))
        target_labels = getattr(self.config, 'target_labels', [])

        for record in samples:
            true_toxic = record['true'].get('toxic')
            pred_toxic = record['pred'].get('toxic')
            if true_toxic is not None and pred_toxic is not None and true_toxic != pred_toxic:
                entry = {
                    'text': record['text'],
                    'true': true_toxic,
                    'pred': pred_toxic
                }
                if true_toxic == 1 and pred_toxic == 0:
                    buckets['toxic_false_negatives'].append(entry)
                elif true_toxic == 0 and pred_toxic == 1:
                    buckets['toxic_false_positives'].append(entry)

            true_expr = record['true'].get('expression')
            pred_expr = record['pred'].get('expression')
            if true_expr is not None and pred_expr is not None and true_expr != pred_expr:
                if isinstance(true_expr, list) and isinstance(pred_expr, list):
                    true_expr_id = int(self._onehot_to_id(np.array(true_expr))[0])
                    pred_expr_id = int(self._onehot_to_id(np.array(pred_expr))[0])
                else:
                    true_expr_id = int(true_expr)
                    pred_expr_id = int(pred_expr)
                key = f"{true_expr_id}->{pred_expr_id}"
                if key not in buckets['expression_confusions']:
                    buckets['expression_confusions'][key] = []
                buckets['expression_confusions'][key].append({
                    'text': record['text'],
                    'true_label': expression_labels[true_expr_id] if true_expr_id < len(expression_labels) else true_expr_id,
                    'pred_label': expression_labels[pred_expr_id] if pred_expr_id < len(expression_labels) else pred_expr_id
                })

            true_target = record['true'].get('target')
            pred_target = record['pred'].get('target')
            if true_target and pred_target and true_target != pred_target:
                diff = {
                    'text': record['text'],
                    'true_active': [target_labels[i] if i < len(target_labels) else i
                                    for i, v in enumerate(true_target) if v == 1],
                    'pred_active': [target_labels[i] if i < len(target_labels) else i
                                    for i, v in enumerate(pred_target) if v == 1]
                }
                buckets['target_mismatches'].append(diff)

        summary = {
            'toxic_false_negatives': buckets['toxic_false_negatives'][:50],
            'toxic_false_positives': buckets['toxic_false_positives'][:50],
            'expression_confusions': {k: v[:30] for k, v in buckets['expression_confusions'].items()},
            'target_mismatches': buckets['target_mismatches'][:50]
        }

        with open(output_dir / 'error_samples.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _save_target_label_stats(self, output_dir: Path, y_true_list: List, y_pred_list: List):
        """."""
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        if y_true.size == 0:
            return

        stats = []
        label_names = getattr(self.config, 'target_labels', [])
        for idx in range(y_true.shape[1]):
            true_col = y_true[:, idx]
            pred_col = y_pred[:, idx]
            tp = int(np.logical_and(true_col == 1, pred_col == 1).sum())
            fp = int(np.logical_and(true_col == 0, pred_col == 1).sum())
            fn = int(np.logical_and(true_col == 1, pred_col == 0).sum())
            tn = int(np.logical_and(true_col == 0, pred_col == 0).sum())
            label_name = label_names[idx] if idx < len(label_names) else str(idx)
            stats.append({
                'label': label_name,
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn
            })

        with open(output_dir / 'target_label_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)





