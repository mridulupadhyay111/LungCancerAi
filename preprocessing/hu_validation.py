"""
hu_validation.py

Research-grade HU validation for NSCLC-Radiomics.

Author: LungCancerAI

Purpose:
--------
Verify that CT voxel values are already in Hounsfield Units (HU)
and generate quality-control statistics.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import SimpleITK as sitk


@dataclass
class HUStatistics:
    minimum: float
    maximum: float
    mean: float
    std: float
    air_percentage: float
    soft_tissue_percentage: float
    bone_percentage: float
    valid_hu: bool


class HUValidator:

    @staticmethod
    def validate(
        image: sitk.Image,
    ) -> tuple[np.ndarray, HUStatistics]:

        volume = sitk.GetArrayFromImage(image).astype(np.float32)

        minimum = float(np.min(volume))
        maximum = float(np.max(volume))
        mean = float(np.mean(volume))
        std = float(np.std(volume))

        total = volume.size

        air = np.sum(volume <= -900)

        soft = np.sum((volume >= -100) & (volume <= 150))

        bone = np.sum(volume >= 300)

        air_percentage = air / total

        soft_percentage = soft / total

        bone_percentage = bone / total

        valid_hu = (
            minimum <= -900
            and maximum >= 500
        )

        stats = HUStatistics(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            air_percentage=air_percentage,
            soft_tissue_percentage=soft_percentage,
            bone_percentage=bone_percentage,
            valid_hu=valid_hu,
        )

        return volume, stats