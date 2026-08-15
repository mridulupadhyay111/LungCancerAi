"""
trainer/metrics.py

Research-grade Metrics

Supports
---------
• Histology Accuracy
• Stage Accuracy
• Precision
• Recall
• F1 Score
• ROC AUC
• Survival MAE
"""

from __future__ import annotations

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class MetricTracker:

    def __init__(self):

        self.reset()

    def reset(self):

        self.histology_pred = []
        self.histology_gt = []

        self.stage_pred = []
        self.stage_gt = []

        self.survival_pred = []
        self.survival_gt = []

    def update(self, outputs, batch):

        histology = torch.argmax(
            outputs["histology_logits"],
            dim=1,
        )

        stage = torch.argmax(
            outputs["stage_logits"],
            dim=1,
        )

        self.histology_pred.extend(
            histology.cpu().numpy()
        )

        self.histology_gt.extend(
            batch["histology"].cpu().numpy()
        )

        self.stage_pred.extend(
            stage.cpu().numpy()
        )

        self.stage_gt.extend(
            batch["stage"].cpu().numpy()
        )

        self.survival_pred.extend(
            outputs["survival_prediction"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        self.survival_gt.extend(
            batch["survival"]
            .cpu()
            .numpy()
            .flatten()
        )

    def compute(self):

        metrics = {}

        metrics["histology_acc"] = accuracy_score(
            self.histology_gt,
            self.histology_pred,
        )

        metrics["stage_acc"] = accuracy_score(
            self.stage_gt,
            self.stage_pred,
        )

        metrics["histology_precision"] = precision_score(
            self.histology_gt,
            self.histology_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["histology_recall"] = recall_score(
            self.histology_gt,
            self.histology_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["histology_f1"] = f1_score(
            self.histology_gt,
            self.histology_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["stage_precision"] = precision_score(
            self.stage_gt,
            self.stage_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["stage_recall"] = recall_score(
            self.stage_gt,
            self.stage_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["stage_f1"] = f1_score(
            self.stage_gt,
            self.stage_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["survival_mae"] = np.mean(
            np.abs(
                np.array(self.survival_gt)
                -
                np.array(self.survival_pred)
            )
        )

        return metrics