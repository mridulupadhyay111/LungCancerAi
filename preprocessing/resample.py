"""
Research-grade isotropic resampling.

Supports:
- CT Images (Linear interpolation)
- Segmentation Masks (Nearest Neighbor)

Author: LungCancerAI
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class ResampleInfo:
    original_spacing: Tuple[float, float, float]
    original_size: Tuple[int, int, int]
    new_spacing: Tuple[float, float, float]
    new_size: Tuple[int, int, int]


class Resampler:

    def __init__(self,
                 target_spacing=(1.0, 1.0, 1.0)):
        self.target_spacing = tuple(target_spacing)

    def _compute_new_size(self,
                          image: sitk.Image):

        old_spacing = image.GetSpacing()
        old_size = image.GetSize()

        new_size = []

        for i in range(3):

            size = int(round(
                old_size[i] *
                old_spacing[i] /
                self.target_spacing[i]
            ))

            new_size.append(size)

        return tuple(new_size)

    def _resample(self,
                  image: sitk.Image,
                  interpolator):

        new_size = self._compute_new_size(image)

        resampler = sitk.ResampleImageFilter()

        resampler.SetInterpolator(interpolator)

        resampler.SetOutputSpacing(
            self.target_spacing
        )

        resampler.SetSize(new_size)

        resampler.SetOutputOrigin(
            image.GetOrigin()
        )

        resampler.SetOutputDirection(
            image.GetDirection()
        )

        resampler.SetTransform(
            sitk.Transform()
        )

        resampler.SetDefaultPixelValue(-1024)

        output = resampler.Execute(image)

        info = ResampleInfo(
            original_spacing=image.GetSpacing(),
            original_size=image.GetSize(),
            new_spacing=output.GetSpacing(),
            new_size=output.GetSize()
        )

        return output, info

    def resample_ct(self,
                    image: sitk.Image):

        return self._resample(
            image,
            sitk.sitkLinear
        )

    def resample_mask(self,
                      mask: sitk.Image):

        return self._resample(
            mask,
            sitk.sitkNearestNeighbor
        )

    @staticmethod
    def validate(image: sitk.Image):

        arr = sitk.GetArrayFromImage(image)

        if np.isnan(arr).any():
            raise ValueError(
                "NaN values detected."
            )

        if np.isinf(arr).any():
            raise ValueError(
                "Infinite values detected."
            )

        return True