"""
Focal Loss Implementation for Multi-Class Classification

Reduces the relative loss for well-classified examples (p > 0.5), putting more
focus on hard, misclassified examples (e.g., underrepresented cancer stages/subtypes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss

    Parameters
    ----------
    alpha : Optional[torch.Tensor]
        Weights for each class. Shape (num_classes,)
    gamma : float
        Focusing parameter. Higher values increase focus on hard examples. Default: 2.0
    label_smoothing : float
        Label smoothing factor. Default: 0.1
    reduction : str
        'mean' or 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Label smoothing target generation
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            if self.label_smoothing > 0:
                smooth_targets = smooth_targets * (
                    1.0 - self.label_smoothing
                ) + self.label_smoothing / num_classes

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Focal factor: (1 - p)^gamma
        p_t = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Cross entropy loss per sample
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_weight = self.alpha[targets]
            focal_loss = focal_loss * alpha_weight

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
