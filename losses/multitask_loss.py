"""
Research-grade Multi-task Loss

Tasks
------
1. Histology Classification
2. Stage Classification
3. Survival Regression

Author
------
LungCancerAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:

    histology_weight: float = 1.0

    stage_weight: float = 1.0

    survival_weight: float = 0.5

    label_smoothing: float = 0.1


class MultiTaskLoss(nn.Module):

    def __init__(

        self,

        config: LossConfig = LossConfig(),

    ):

        super().__init__()

        self.config = config

        self.histology_loss = nn.CrossEntropyLoss(

            label_smoothing=config.label_smoothing

        )

        self.stage_loss = nn.CrossEntropyLoss(

            label_smoothing=config.label_smoothing

        )

        self.survival_loss = nn.SmoothL1Loss()

    #########################################################

    def forward(

        self,

        outputs,

        targets,

    ):

        histology = self.histology_loss(

            outputs["histology_logits"],

            targets["histology"],

        )

        stage = self.stage_loss(

            outputs["stage_logits"],

            targets["stage"],

        )

        survival = self.survival_loss(

            outputs["survival_prediction"].squeeze(1),

            targets["survival"],

        )

        total = (

            self.config.histology_weight * histology

            +

            self.config.stage_weight * stage

            +

            self.config.survival_weight * survival

        )

        return {

            "loss": total,

            "histology_loss": histology,

            "stage_loss": stage,

            "survival_loss": survival,

        }
    def build_targets(batch):

     return {

        "histology": batch["histology"],

        "stage": batch["stage"],

        "survival": batch["survival"],

    }    