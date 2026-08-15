"""
metrics.py

Research Grade Metrics

Author
------
LungCancerAI
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

)


class ClassificationMetrics:

    @staticmethod
    def compute(

        prediction,

        target,

    ):

        prediction = prediction.cpu().numpy()

        target = target.cpu().numpy()

        return {

            "accuracy":

                accuracy_score(

                    target,

                    prediction,

                ),

            "precision":

                precision_score(

                    target,

                    prediction,

                    average="macro",

                    zero_division=0,

                ),

            "recall":

                recall_score(

                    target,

                    prediction,

                    average="macro",

                    zero_division=0,

                ),

            "f1":

                f1_score(

                    target,

                    prediction,

                    average="macro",

                    zero_division=0,

                ),

        }


class RunningAverage:

    def __init__(self):

        self.reset()

    def reset(self):

        self.total = 0

        self.count = 0

    def update(

        self,

        value,

        n=1,

    ):

        self.total += value * n

        self.count += n

    @property
    def average(self):

        if self.count == 0:

            return 0

        return self.total / self.count


class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0

    def update(

        self,

        val,

        n=1,

    ):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count


def topk_accuracy(

    logits,

    labels,

    k=1,

):

    _, pred = logits.topk(

        k,

        dim=1,

    )

    pred = pred.t()

    correct = pred.eq(

        labels.view(1, -1)

    )

    correct = correct.reshape(-1).float().sum()

    return correct / labels.size(0)


def print_metrics(metrics):

    print("=" * 60)

    for k, v in metrics.items():

        print(f"{k:20s}: {v:.4f}")

    print("=" * 60)