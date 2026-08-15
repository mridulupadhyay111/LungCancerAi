"""Normalize placeholder"""
"""
normalization.py

Research-grade intensity normalization for CT volumes.

Author: LungCancerAI
"""

from enum import Enum
import numpy as np


class NormalizationType(Enum):

    Z_SCORE = "zscore"

    MIN_MAX = "minmax"


class Normalizer:

    def __init__(self,
                 method=NormalizationType.Z_SCORE):

        self.method = method

    def apply(self,
              volume: np.ndarray):

        volume = volume.astype(np.float32)

        if self.method == NormalizationType.Z_SCORE:

            mean = volume.mean()

            std = volume.std()

            if std < 1e-8:
                std = 1e-8

            volume = (volume - mean) / std

        elif self.method == NormalizationType.MIN_MAX:

            minimum = volume.min()

            maximum = volume.max()

            volume = (volume - minimum) / (
                maximum - minimum + 1e-8
            )

        else:

            raise ValueError(
                "Unsupported normalization."
            )

        return volume

    def summary(self, volume):

        print("=" * 60)

        print("Normalization Summary")

        print("=" * 60)

        print("Minimum :", volume.min())

        print("Maximum :", volume.max())

        print("Mean    :", volume.mean())

        print("Std     :", volume.std())

        print("=" * 60)