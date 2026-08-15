"""
Research Grade Evaluation Metrics
Supports:
- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix
- ROC AUC
- Survival MAE
"""

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
)


class Evaluator:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    def classification_metrics(
        self,
        prediction,
        target,
    ):

        prediction = np.asarray(prediction)
        target = np.asarray(target)

        metrics = {}

        metrics["accuracy"] = accuracy_score(
            target,
            prediction,
        )

        metrics["precision"] = precision_score(
            target,
            prediction,
            average="macro",
            zero_division=0,
        )

        metrics["recall"] = recall_score(
            target,
            prediction,
            average="macro",
            zero_division=0,
        )

        metrics["f1"] = f1_score(
            target,
            prediction,
            average="macro",
            zero_division=0,
        )

        metrics["confusion_matrix"] = confusion_matrix(
            target,
            prediction,
        )

        metrics["classification_report"] = classification_report(
            target,
            prediction,
            zero_division=0,
        )

        return metrics

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    def roc_auc(
        self,
        target,
        probabilities,
    ):

        try:

            score = roc_auc_score(

                target,

                probabilities,

                multi_class="ovr",

            )

        except Exception:

            score = 0.0

        return score

    # --------------------------------------------------------
    # Survival MAE
    # --------------------------------------------------------

    def survival_metrics(
        self,
        prediction,
        target,
    ):

        prediction = np.asarray(prediction)

        target = np.asarray(target)

        return {

            "mae": mean_absolute_error(

                target,

                prediction,

            )

        }

    # --------------------------------------------------------
    # Complete Evaluation
    # --------------------------------------------------------

    def evaluate(

        self,

        outputs,

        targets,

    ):

        histology_pred = outputs["histology_logits"].argmax(1).cpu().numpy()

        stage_pred = outputs["stage_logits"].argmax(1).cpu().numpy()

        survival_pred = outputs["survival_prediction"].squeeze().detach().cpu().numpy()

        result = {

            "histology":

                self.classification_metrics(

                    histology_pred,

                    targets["histology"].cpu().numpy(),

                ),

            "stage":

                self.classification_metrics(

                    stage_pred,

                    targets["stage"].cpu().numpy(),

                ),

            "survival":

                self.survival_metrics(

                    survival_pred,

                    targets["survival"].cpu().numpy(),

                ),

        }

        return result