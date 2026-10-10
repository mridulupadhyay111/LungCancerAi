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


class RandomAugment3D:
    """
    3D Medical Augmentation Pipeline for CT Tumor Volumes.
    Includes random rotations, flips, intensity scaling, and noise.
    """
    def __init__(self, training=True):
        self.training = training

    def __call__(self, image, mask=None):
        if not self.training:
            return image, mask

        # Random Flip along spatial axes (D, H, W)
        for axis in (1, 2, 3):
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[axis])
                if mask is not None:
                    mask = torch.flip(mask, dims=[axis])

        # Random 90-degree rotations in spatial plane (H, W)
        if torch.rand(1).item() > 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k=k, dims=[2, 3])
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=[2, 3])

        # Random Intensity Scaling (Contrast / Brightness)
        if torch.rand(1).item() > 0.5:
            scale = 0.85 + torch.rand(1).item() * 0.3
            image = image * scale

        # Random Additive Gaussian Noise
        if torch.rand(1).item() > 0.3:
            noise = torch.randn_like(image) * 0.02
            image = image + noise

        return image, mask
