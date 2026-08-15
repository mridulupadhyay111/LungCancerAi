"""Transforms placeholder"""
import numpy as np
import torch
import torch.nn.functional as F


class Resize3D:

    def __init__(self, target_size=(128, 128, 128)):
        self.target_size = target_size

    def __call__(self, image):

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        )

        return image.squeeze(0).squeeze(0)


class MaskResize3D:

    def __init__(self, target_size=(128, 128, 128)):
        self.target_size = target_size

    def __call__(self, mask):

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.float()

        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        mask = mask.unsqueeze(0)

        mask = F.interpolate(
            mask,
            size=self.target_size,
            mode="nearest",
        )

        return mask.squeeze(0).squeeze(0)