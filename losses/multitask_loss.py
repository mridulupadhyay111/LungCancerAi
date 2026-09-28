"""
Research-grade Dual-Classification Loss (Histology & Stage Focus)

Tasks:
1. Histology Classification (Categorical)
2. Stage Classification (TNM Stage Categorical)
3. Survival Regression (Weight set to 0.0 to prevent gradient interference)

Author: LungCancerAI
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from losses.focal_loss import FocalLoss


@dataclass
class LossConfig:
    histology_weight: float = 1.0
    stage_weight: float = 1.0
    survival_weight: float = 0.0  # Zero out survival loss to eliminate gradient noise
    label_smoothing: float = 0.1
    use_focal_loss: bool = True
    gamma: float = 2.0


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        config: LossConfig = LossConfig(),
        histology_class_weights: Optional[torch.Tensor] = None,
        stage_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config

        if config.use_focal_loss:
            self.histology_loss = FocalLoss(
                alpha=histology_class_weights,
                gamma=config.gamma,
                label_smoothing=config.label_smoothing,
            )
            self.stage_loss = FocalLoss(
                alpha=stage_class_weights,
                gamma=config.gamma,
                label_smoothing=config.label_smoothing,
            )
        else:
            self.histology_loss = nn.CrossEntropyLoss(
                weight=histology_class_weights,
                label_smoothing=config.label_smoothing,
            )
            self.stage_loss = nn.CrossEntropyLoss(
                weight=stage_class_weights,
                label_smoothing=config.label_smoothing,
            )

        self.survival_loss = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        histology = self.histology_loss(
            outputs["histology_logits"],
            targets["histology"],
        )

        stage = self.stage_loss(
            outputs["stage_logits"],
            targets["stage"],
        )

        # Compute survival loss safely for logging/UI fallback without backpropagating it
        if "survival_prediction" in outputs and "survival" in targets and self.config.survival_weight > 0:
            survival = self.survival_loss(
                outputs["survival_prediction"].squeeze(-1),
                targets["survival"],
            )
        else:
            survival = torch.tensor(0.0, device=histology.device)

        total = (
            self.config.histology_weight * histology
            + self.config.stage_weight * stage
            + self.config.survival_weight * survival
        )

        return {
            "loss": total,
            "histology_loss": histology,
            "stage_loss": stage,
            "survival_loss": survival,
        }

    @staticmethod
    def build_targets(batch):
        return {
            "histology": batch["histology"],
            "stage": batch["stage"],
            "survival": batch["survival"],
        }