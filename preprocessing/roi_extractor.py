"""
Research-grade ROI extraction.

Extracts a tumor-centered region of interest from a CT volume and its
corresponding binary mask.

Author: LungCancerAI
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BoundingBox:

    z_min: int
    z_max: int

    y_min: int
    y_max: int

    x_min: int
    x_max: int


class ROIExtractor:

    def __init__(self, margin: int = 20):

        self.margin = margin

    def compute_bbox(self,
                     mask: np.ndarray) -> BoundingBox:

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if np.sum(mask) == 0:
            raise ValueError("Empty tumor mask.")

        z, y, x = np.where(mask > 0)

        z_min = z.min()
        z_max = z.max()

        y_min = y.min()
        y_max = y.max()

        x_min = x.min()
        x_max = x.max()

        return BoundingBox(
            z_min,
            z_max,
            y_min,
            y_max,
            x_min,
            x_max
        )

    def expand_bbox(self,
                    bbox: BoundingBox,
                    shape: Tuple[int, int, int]):

        z = shape[0]
        y = shape[1]
        x = shape[2]

        return BoundingBox(

            max(0, bbox.z_min - self.margin),
            min(z, bbox.z_max + self.margin),

            max(0, bbox.y_min - self.margin),
            min(y, bbox.y_max + self.margin),

            max(0, bbox.x_min - self.margin),
            min(x, bbox.x_max + self.margin)

        )

    def crop(self,
             image: np.ndarray,
             mask: np.ndarray):

        bbox = self.compute_bbox(mask)

        bbox = self.expand_bbox(
            bbox,
            mask.shape
        )

        cropped_image = image[
            bbox.z_min:bbox.z_max,
            bbox.y_min:bbox.y_max,
            bbox.x_min:bbox.x_max
        ]

        cropped_mask = mask[
            bbox.z_min:bbox.z_max,
            bbox.y_min:bbox.y_max,
            bbox.x_min:bbox.x_max
        ]

        if cropped_image.shape != cropped_mask.shape:
            raise RuntimeError(
                "Shape mismatch after crop."
            )

        if np.sum(cropped_mask) == 0:
            raise RuntimeError(
                "Tumor disappeared after crop."
            )

        return cropped_image, cropped_mask, bbox