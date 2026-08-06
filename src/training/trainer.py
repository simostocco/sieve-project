"""
Training loop for SIEVE models.

This module implements the training infrastructure including:
- Training and validation loops
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Metrics tracking

Author: Francesco Lescai
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import yaml

from .loss import SIEVELoss


class Trainer:
    """
    Trainer for SIEVE models.

    Handles training loop, validation, checkpointing, and early stopping.

    Args:
        model: SIEVE model to train
        optimizer: PyTorch optimizer
        loss_fn: Loss function (SIEVELoss instance)
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Number of epochs without improvement before stopping (default: 10)
        gradient_clip_value: Maximum gradient norm for clipping (default: None)

    Attributes:
        model: The model being trained
        optimizer: The optimizer
        loss_fn: Loss function
        device: Training device
        checkpoint_dir: Path to checkpoint directory
        scheduler: Learning rate scheduler
        early_stopping_patience: Early stopping patience
        gradient_clip_value: Gradient clipping value
        history: Training history (losses, metrics per epoch)
        best_val_auc: Best validation AUC achieved
        epochs_without_improvement: Counter for early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: SIEVELoss,
        device: str,
        checkpoint_dir: Path,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping_patience: int = 10,
        gradient_clip_value: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        if checkpoint_metadata is not None and not isinstance(checkpoint_metadata, dict):
            raise ValueError("checkpoint_metadata must be a dict when provided")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_metadata = (
            copy.deepcopy(checkpoint_metadata)
            if checkpoint_metadata is not None
            else None
        )

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_classification_loss': [],
            'train_attribution_loss': [],
            'train_auc': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_classification_loss': [],
            'val_attribution_loss': [],
            'val_auc': [],
            'val_accuracy': [],
            'learning_rate': [],
        }
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics (loss, AUC, accuracy)
        """
        self.model.train()

        all_losses = []
        all_classification_losses = []
        all_attribution_losses = []
        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()  # Zero once at start

        for batch_idx, batch in enumerate(train_loader):
            # Check if model has train_step (for chunked processing)
            if hasattr(self.model, 'train_step'):
                # Use model's train_step for chunked processing
                loss_output, logits = self.model.train_step(batch, self.loss_fn, self.device)

                # Build loss_dict from the returned output
                if isinstance(loss_output, dict):
                    loss_dict = loss_output
                    loss = loss_output['total']
                else:
                    # Scalar loss (e.g. BCEWithLogitsLoss), no decomposition available
                    loss = loss_output
                    loss_dict = {
                        'total': loss,
                        'classification': loss,
                        'attribution_sparsity': torch.tensor(0.0, device=loss.device),
                    }

                # Get sample labels for metrics
                if 'original_sample_indices' in batch:
                    # Chunked: labels are per-sample, need to aggregate from chunks
                    # CRITICAL: Move to device BEFORE processing
                    original_indices = batch['original_sample_indices'].to(self.device)
                    batch_labels = batch['labels'].to(self.device)
                    # CRITICAL: Must use sorted=True to match train_step's prediction ordering
                    unique_samples = original_indices.unique(sorted=True)
                    labels_for_metrics = torch.zeros(len(unique_samples), dtype=torch.long, device=self.device)
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_indices == sample_idx)
                        labels_for_metrics[i] = batch_labels[chunk_mask][0]
                    labels = labels_for_metrics
                else:
                    labels = batch['labels']
            else:
                # Standard processing (no chunking)
                features = batch.get('features')
                if features is not None:
                    features = features.to(self.device)
                content_features = batch.get('content_features')
                absolute_position_features = batch.get('absolute_position_features')
                if content_features is not None:
                    content_features = content_features.to(self.device)
                if absolute_position_features is not None:
                    absolute_position_features = absolute_position_features.to(self.device)
                split_feature_kwargs = {}
                if content_features is not None or absolute_position_features is not None:
                    # Only send the new keywords when a split tensor is present;
                    # older model doubles used in tests may only accept the
                    # historical feature signature.
                    split_feature_kwargs = {
                        'content_features': content_features,
                        'absolute_position_features': absolute_position_features,
                    }
                positions = batch['positions'].to(self.device)
                gene_ids = batch['gene_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chrom_ids = batch.get('chrom_ids')
                if chrom_ids is not None:
                    chrom_ids = chrom_ids.to(self.device)

                # Build covariates if available
                covariates = None
                batch_sex = batch.get('sex')
                num_cov = getattr(self.model, 'num_covariates', 0)
                if num_cov > 0 and batch_sex is not None:
                    batch_sex = batch_sex.to(self.device)
                    batch_size = batch_sex.shape[0]
                    covariates = torch.zeros(
                        batch_size, num_cov,
                        device=self.device, dtype=batch_sex.dtype
                    )
                    covariates[:, 0] = batch_sex

                # Forward pass
                if self.loss_fn.lambda_attr > 0:
                    # Need variant embeddings for attribution loss
                    logits, intermediates = self.model(
                        features, positions, gene_ids, mask,
                        covariates=covariates,
                        return_intermediate=True,
                        chrom_ids=chrom_ids,
                        **split_feature_kwargs,
                    )
                    variant_embeddings = intermediates['variant_embeddings']
                    loss_dict = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        variant_embeddings=variant_embeddings,
                        mask=mask,
                    )
                else:
                    # Standard forward pass
                    logits, _ = self.model(
                        features, positions, gene_ids, mask,
                        covariates=covariates,
                        chrom_ids=chrom_ids,
                        **split_feature_kwargs,
                    )
                    loss_dict = self.loss_fn(logits=logits, labels=labels)

                loss = loss_dict['total']

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaled_loss.backward()

            # Optimizer step every accumulation_steps batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_value
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track metrics (use unscaled loss)
            all_losses.append(loss.item())
            all_classification_losses.append(loss_dict['classification'].item())
            all_attribution_losses.append(loss_dict['attribution_sparsity'].item())

            # Store predictions and labels for AUC
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        # Final optimizer step if there are remaining gradients
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Compute epoch metrics
        train_loss = np.mean(all_losses)
        train_classification_loss = np.mean(all_classification_losses)
        train_attribution_loss = np.mean(all_attribution_losses)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)

        return {
            'loss': train_loss,
            'classification_loss': train_classification_loss,
            'attribution_loss': train_attribution_loss,
            'auc': train_auc,
            'accuracy': train_accuracy,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics (loss, AUC, accuracy)
        """
        self.model.eval()

        all_losses = []
        all_classification_losses = []
        all_attribution_losses = []
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(val_loader):
            # Check if model has train_step (for chunked processing)
            if hasattr(self.model, 'train_step'):
                # Use model's train_step for chunked processing (works for eval too)
                loss_output, logits = self.model.train_step(batch, self.loss_fn, self.device)

                if isinstance(loss_output, dict):
                    loss_dict = loss_output
                    loss = loss_output['total']
                else:
                    loss = loss_output
                    loss_dict = {
                        'total': loss,
                        'classification': loss,
                        'attribution_sparsity': torch.tensor(0.0, device=loss.device),
                    }

                # Get sample labels for metrics
                if 'original_sample_indices' in batch:
                    # Chunked: labels are per-sample, need to aggregate from chunks
                    # CRITICAL: Move to device BEFORE processing
                    original_indices = batch['original_sample_indices'].to(self.device)
                    batch_labels = batch['labels'].to(self.device)
                    # CRITICAL: Must use sorted=True to match train_step's prediction ordering
                    unique_samples = original_indices.unique(sorted=True)
                    labels_for_metrics = torch.zeros(len(unique_samples), dtype=torch.long, device=self.device)
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_indices == sample_idx)
                        labels_for_metrics[i] = batch_labels[chunk_mask][0]
                    labels = labels_for_metrics
                else:
                    labels = batch['labels']
            else:
                # Standard processing (no chunking)
                features = batch.get('features')
                if features is not None:
                    features = features.to(self.device)
                content_features = batch.get('content_features')
                absolute_position_features = batch.get('absolute_position_features')
                if content_features is not None:
                    content_features = content_features.to(self.device)
                if absolute_position_features is not None:
                    absolute_position_features = absolute_position_features.to(self.device)
                split_feature_kwargs = {}
                if content_features is not None or absolute_position_features is not None:
                    # Keep legacy-only batches and old model doubles on the
                    # historical call signature while dataset split batches use
                    # the model-side composer.
                    split_feature_kwargs = {
                        'content_features': content_features,
                        'absolute_position_features': absolute_position_features,
                    }
                positions = batch['positions'].to(self.device)
                gene_ids = batch['gene_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chrom_ids = batch.get('chrom_ids')
                if chrom_ids is not None:
                    chrom_ids = chrom_ids.to(self.device)

                # Build covariates if available
                covariates = None
                batch_sex = batch.get('sex')
                num_cov = getattr(self.model, 'num_covariates', 0)
                if num_cov > 0 and batch_sex is not None:
                    batch_sex = batch_sex.to(self.device)
                    batch_size = batch_sex.shape[0]
                    covariates = torch.zeros(
                        batch_size, num_cov,
                        device=self.device, dtype=batch_sex.dtype
                    )
                    covariates[:, 0] = batch_sex

                # Forward pass
                if self.loss_fn.lambda_attr > 0:
                    logits, intermediates = self.model(
                        features, positions, gene_ids, mask,
                        covariates=covariates,
                        return_intermediate=True,
                        chrom_ids=chrom_ids,
                        **split_feature_kwargs,
                    )
                    variant_embeddings = intermediates['variant_embeddings']
                    loss_dict = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        variant_embeddings=variant_embeddings,
                        mask=mask,
                    )
                else:
                    logits, _ = self.model(
                        features, positions, gene_ids, mask,
                        covariates=covariates,
                        chrom_ids=chrom_ids,
                        **split_feature_kwargs,
                    )
                    loss_dict = self.loss_fn(logits=logits, labels=labels)

                loss = loss_dict['total']

            # Track metrics
            all_losses.append(loss.item())
            all_classification_losses.append(loss_dict['classification'].item())
            all_attribution_losses.append(loss_dict['attribution_sparsity'].item())

            # Store predictions and labels for AUC
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        val_loss = np.mean(all_losses)
        val_classification_loss = np.mean(all_classification_losses)
        val_attribution_loss = np.mean(all_attribution_losses)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)

        return {
            'loss': val_loss,
            'classification_loss': val_classification_loss,
            'attribution_loss': val_attribution_loss,
            'auc': val_auc,
            'accuracy': val_accuracy,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            verbose: Whether to print progress (default: True)

        Returns:
            Training history dictionary

        Raises:
            KeyboardInterrupt: If training is interrupted by user
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_classification_loss'].append(train_metrics['classification_loss'])
            self.history['train_attribution_loss'].append(train_metrics['attribution_loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_classification_loss'].append(val_metrics['classification_loss'])
            self.history['val_attribution_loss'].append(val_metrics['attribution_loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"AUC: {train_metrics['auc']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
                if self.loss_fn.lambda_attr > 0:
                    print(f"  Attribution Loss - Train: {train_metrics['attribution_loss']:.4f}, "
                          f"Val: {val_metrics['attribution_loss']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['auc'])

            # Checkpointing
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', val_metrics)
                if verbose:
                    print(f"  → New best model (AUC: {self.best_val_auc:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Save last checkpoint
            self.save_checkpoint('last_model.pt', val_metrics)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch+1} epochs "
                          f"({self.early_stopping_patience} epochs without improvement)")
                break

        # Save training history
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Current metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_auc': self.best_val_auc,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.checkpoint_metadata is not None:
            checkpoint['metadata'] = copy.deepcopy(self.checkpoint_metadata)

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def save_history(self, filename: str = 'training_history.yaml') -> None:
        """
        Save training history to YAML file.

        Args:
            filename: History filename (default: 'training_history.yaml')
        """
        history_path = self.checkpoint_dir / filename

        # Convert history to serializable format
        history_serializable = {
            key: [float(val) for val in values]
            for key, values in self.history.items()
        }

        # Add metadata
        history_data = {
            'history': history_serializable,
            'best_val_auc': float(self.best_val_auc),
            'total_epochs': len(self.history['train_loss']),
            'best_epoch': int(np.argmax(self.history['val_auc'])) + 1 if self.history['val_auc'] else 0,
        }

        with open(history_path, 'w') as f:
            yaml.dump(history_data, f, default_flow_style=False)

    def load_checkpoint(self, filename: str) -> Dict[str, float]:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Metrics from the loaded checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False  # We save metrics and history, not just weights
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['metrics']
