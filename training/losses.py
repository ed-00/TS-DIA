#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Loss Functions for Training

This module provides a comprehensive set of loss functions including:
- Standard losses (cross entropy, MSE, BCE, etc.)
- Advanced losses (focal loss, label smoothing, etc.)
- Auxiliary losses (regularization, contrastive, etc.)
- Custom loss composition

Key Functions:
    create_loss_function: Factory for creating loss functions
    compute_loss: Compute total loss with auxiliary losses
    LossRegistry: Registry of available loss functions
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Label smoothing helps prevent overconfidence and improves generalization.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)

        # One-hot encoding with smoothing
        num_classes = inputs.size(-1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            -1, targets.unsqueeze(-1), 1.0
        )
        targets_one_hot = (
            targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        )

        loss = -(targets_one_hot * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class NormRegularizationLoss(nn.Module):
    """
    L1 or L2 norm regularization on model parameters.
    """

    def __init__(self, norm_type: str = "l2"):
        super().__init__()
        self.norm_type = norm_type

    def forward(self, model: nn.Module) -> torch.Tensor:
        norm = 0.0
        for param in model.parameters():
            if self.norm_type == "l1":
                norm += torch.abs(param).sum()
            elif self.norm_type == "l2":
                norm += torch.square(param).sum()
        return norm


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation learning.

    Useful for auxiliary tasks in self-supervised or multi-task learning.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: Normalized feature embeddings [batch_size, feature_dim]
            labels: Optional labels for supervised contrastive learning
        """
        # Normalize features
        features = F.normalize(features, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs
        batch_size = features.size(0)
        if labels is not None:
            # Supervised contrastive: same label = positive pair
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
            # Remove diagonal (self-similarity)
            mask.fill_diagonal_(0)
        else:
            # Self-supervised contrastive: adjacent samples are positive pairs
            mask = torch.eye(batch_size, device=features.device)
            mask = torch.roll(mask, 1, dims=0)

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LossRegistry:
    """Registry of available loss functions."""

    _losses = {
        # Standard losses
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "nll": nn.NLLLoss,
        "kl_div": nn.KLDivLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "huber": nn.HuberLoss,
        # Advanced losses
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingCrossEntropy,
        # Auxiliary losses
        "norm_reg": NormRegularizationLoss,
        "contrastive": ContrastiveLoss,
    }

    @classmethod
    def get(cls, loss_name: str, **kwargs) -> nn.Module:
        """Get loss function by name."""
        if loss_name not in cls._losses:
            available = ", ".join(cls._losses.keys())
            raise ValueError(
                f"Unknown loss function: {loss_name}. Available: {available}"
            )
        return cls._losses[loss_name](**kwargs)

    @classmethod
    def register(cls, name: str, loss_class: type):
        """Register a custom loss function."""
        cls._losses[name] = loss_class

    @classmethod
    def list_available(cls) -> list:
        """List all available loss functions."""
        return list(cls._losses.keys())


def create_loss_function(loss_config: LossConfig) -> nn.Module:
    """
    Create main loss function from configuration.

    Args:
        loss_config: Loss configuration

    Returns:
        Loss function module

    Example:
        ```python
        loss_config = LossConfig(
            main="cross_entropy",
            label_smoothing=0.1,
            reduction="mean"
        )
        loss_fn = create_loss_function(loss_config)
        ```
    """
    loss_kwargs = {"reduction": loss_config.reduction}

    # Add loss-specific kwargs
    if loss_config.label_smoothing is not None:
        if loss_config.main == "cross_entropy":
            # Use label smoothing version
            return LabelSmoothingCrossEntropy(
                smoothing=loss_config.label_smoothing,
                reduction=loss_config.reduction,
            )
        else:
            loss_kwargs["label_smoothing"] = loss_config.label_smoothing

    if loss_config.focal_alpha is not None or loss_config.focal_gamma is not None:
        if loss_config.main == "focal":
            loss_kwargs["alpha"] = loss_config.focal_alpha
            loss_kwargs["gamma"] = loss_config.focal_gamma or 2.0

    return LossRegistry.get(loss_config.main, **loss_kwargs)


def create_auxiliary_losses(loss_config: LossConfig) -> Dict[str, nn.Module]:
    """
    Create auxiliary loss functions from configuration.

    Args:
        loss_config: Loss configuration

    Returns:
        Dictionary mapping loss names to loss modules

    Example:
        ```python
        loss_config = LossConfig(
            main="cross_entropy",
            auxiliary={
                "norm_reg": 0.1,
                "contrastive": 0.2
            }
        )
        aux_losses = create_auxiliary_losses(loss_config)
        ```
    """
    if not loss_config.auxiliary:
        return {}

    aux_losses = {}
    for loss_name in loss_config.auxiliary.keys():
        aux_losses[loss_name] = LossRegistry.get(loss_name)

    return aux_losses


def compute_loss(
    main_loss_fn: nn.Module,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    auxiliary_losses: Optional[Dict[str, nn.Module]] = None,
    auxiliary_weights: Optional[Dict[str, float]] = None,
    model: Optional[nn.Module] = None,
    **aux_loss_kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute total loss including main and auxiliary losses.

    Args:
        main_loss_fn: Main loss function
        outputs: Model outputs
        targets: Target labels/values
        auxiliary_losses: Dictionary of auxiliary loss functions
        auxiliary_weights: Dictionary of auxiliary loss weights
        model: Model (needed for some auxiliary losses like norm_reg)
        **aux_loss_kwargs: Additional kwargs for auxiliary losses

    Returns:
        Dictionary with 'total', 'main', and individual auxiliary losses

    Example:
        ```python
        loss_dict = compute_loss(
            main_loss_fn=cross_entropy,
            outputs=model_outputs,
            targets=labels,
            auxiliary_losses={"norm_reg": norm_reg_loss},
            auxiliary_weights={"norm_reg": 0.1},
            model=model
        )
        total_loss = loss_dict['total']
        ```
    """
    # Compute main loss
    main_loss = main_loss_fn(outputs, targets)

    loss_dict = {
        "main": main_loss,
        "total": main_loss,
    }

    # Compute auxiliary losses
    if auxiliary_losses and auxiliary_weights:
        for loss_name, aux_loss_fn in auxiliary_losses.items():
            weight = auxiliary_weights.get(loss_name, 0.0)

            # Compute auxiliary loss based on type
            if loss_name == "norm_reg":
                if model is None:
                    raise ValueError("Model required for norm_reg loss")
                aux_loss = aux_loss_fn(model)
            elif loss_name == "contrastive":
                # Expect features in aux_loss_kwargs
                features = aux_loss_kwargs.get("features")
                if features is None:
                    raise ValueError("Features required for contrastive loss")
                aux_loss = aux_loss_fn(features, targets)
            else:
                # Generic auxiliary loss
                aux_loss = aux_loss_fn(outputs, targets)

            loss_dict[loss_name] = aux_loss
            loss_dict["total"] = loss_dict["total"] + weight * aux_loss

    return loss_dict


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = "classification",
) -> Dict[str, float]:
    """
    Compute evaluation metrics based on task type.

    Args:
        outputs: Model outputs
        targets: Target labels/values
        task_type: Type of task (classification, regression, etc.)

    Returns:
        Dictionary of metrics

    Example:
        ```python
        metrics = compute_metrics(outputs, targets, task_type="classification")
        accuracy = metrics['accuracy']
        ```
    """
    metrics = {}

    if task_type == "classification":
        # Accuracy
        predictions = outputs.argmax(dim=-1)
        correct = (predictions == targets).float().sum()
        total = targets.size(0)
        metrics["accuracy"] = (correct / total).item()

        # Top-5 accuracy (if applicable)
        if outputs.size(-1) >= 5:
            top5_predictions = outputs.topk(5, dim=-1)[1]
            top5_correct = (
                (top5_predictions == targets.unsqueeze(-1)).any(dim=-1).float().sum()
            )
            metrics["top5_accuracy"] = (top5_correct / total).item()

    elif task_type == "regression":
        # MSE and MAE
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        metrics["mse"] = mse.item()
        metrics["mae"] = mae.item()
        metrics["rmse"] = (mse**0.5).item()

    return metrics
