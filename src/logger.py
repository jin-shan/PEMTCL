# -*- coding: utf-8 -*-
"""

"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class PEMTCLLogger:
    """Training logger for PEMTCL."""

    def __init__(self, log_dir: str, project_name: str):
        self.log_dir = log_dir
        self.project_name = project_name

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 
        self.training_log_file = os.path.join(log_dir, f"training_{self.timestamp}.log")
        self.metrics_log_file = os.path.join(log_dir, f"metrics_{self.timestamp}.json")
        self.config_log_file = os.path.join(log_dir, f"config_{self.timestamp}.json")

        # logger
        self.logger = self._setup_logger()

        # 
        self.training_history = []
        self.best_metrics = {}

    def _setup_logger(self):
        """."""
        logger = logging.getLogger(self.project_name)
        logger.setLevel(logging.INFO)

        # handlers
        logger.handlers.clear()

        # handler
        file_handler = logging.FileHandler(self.training_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # andler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_config(self, config: Dict[str, Any]):
        """"""
        self.logger.info("="*80)
        self.logger.info(f"? {self.project_name}")
        self.logger.info(f"? {self.timestamp}")
        self.logger.info("="*80)

        # 
        import json
        with open(self.config_log_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")

    def log_epoch_start(self, epoch: int, num_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
        self.logger.info(f"{'='*60}")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """epoch"""
        epoch_record = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_history.append(epoch_record)

        # SON
        with open(self.metrics_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

        # metrics
        self.logger.info(f"Epoch {epoch+1} Results:")
        for task_name in ['toxic', 'toxic_type', 'target', 'expression']:
            if f'{task_name}_loss' in metrics:
                self.logger.info(f"  {task_name.upper()}:")
                self.logger.info(f"    Loss: {metrics[f'{task_name}_loss']:.4f}")
                if f'{task_name}_f1' in metrics:
                    self.logger.info(f"    F1: {metrics[f'{task_name}_f1']:.4f}")
                    self.logger.info(f"    Precision: {metrics[f'{task_name}_precision']:.4f}")
                    self.logger.info(f"    Recall: {metrics[f'{task_name}_recall']:.4f}")

        if 'total_loss' in metrics:
            self.logger.info(f"\n  Total Loss: {metrics['total_loss']:.4f}")

        contrastive_keys = [k for k in metrics if k.startswith('contrastive_loss_')]
        if contrastive_keys:
            self.logger.info("  Contrastive:")
            for key in sorted(contrastive_keys):
                task_name = key.replace('contrastive_loss_', '').upper()
                self.logger.info(f"    {task_name} Loss: {metrics[key]:.4f}")
            if 'contrastive_total_loss' in metrics:
                self.logger.info(f"    TOTAL Loss: {metrics['contrastive_total_loss']:.4f}")

        queue_keys = [k for k in metrics if k.startswith('queue_fill_')]
        if queue_keys:
            self.logger.info("  Queue Fill Ratios:")
            for key in sorted(queue_keys):
                task_name = key.replace('queue_fill_', '').upper()
                self.logger.info(f"    {task_name}: {metrics[key]:.3f}")

        diag_keys = [k for k in metrics if k.startswith('neg_')]
        if diag_keys:
            self.logger.info("  Contrastive Diagnostics:")
            for key in sorted(diag_keys):
                self.logger.info(f"    {key}: {metrics[key]:.4f}")

        self._update_best_metrics(metrics)

    def log_training_step(self, step: int, loss: float, learning_rate: float):
        """"""
        if step % 50 == 0:
            self.logger.info(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.2e}")

    def log_evaluation(self, phase: str, metrics: Dict[str, float]):
        """"""
        self.logger.info(f"\n{phase} Results:")
        for task_name in ['toxic', 'toxic_type', 'target', 'expression']:
            if f'{task_name}_f1' in metrics:
                self.logger.info(f"  {task_name.upper()}: F1={metrics[f'{task_name}_f1']:.4f}")

    def log_best_model(self, epoch: int, metrics: Dict[str, float]):
        """."""
        self.logger.info(f"\n{'!'*80}")
        self.logger.info(f"New best model at epoch {epoch+1}!")
        self.logger.info(f"Macro F1: {metrics.get('macro_f1', 0):.4f}")
        self.logger.info(f"{'!'*80}")

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """"""
        self.logger.error(message)
        if exception:
            self.logger.exception(exception)

    def _update_best_metrics(self, metrics: Dict[str, float]):
        """."""
        for key, value in metrics.items():
            if 'f1' in key.lower():
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = self.best_metrics.get(key, value)

    def get_summary(self) -> Dict[str, Any]:
        """"""
        return {
            'project_name': self.project_name,
            'timestamp': self.timestamp,
            'total_epochs': len(self.training_history),
            'best_metrics': self.best_metrics,
            'final_metrics': self.training_history[-1] if self.training_history else {},
            'log_files': {
                'training': self.training_log_file,
                'metrics': self.metrics_log_file,
                'config': self.config_log_file
            }
        }




