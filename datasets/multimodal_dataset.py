"""
multimodal_dataset.py

Research-grade PyTorch Dataset

Loads:
    CT volume
    Tumor mask
    Clinical features
    Labels

Returns

{
    image,
    mask,
    clinical,
    histology,
    stage,
    survival,
    event,
    patient_id
}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.config import Config
from datasets.transforms import (
    Resize3D,
    MaskResize3D,
)


class MultiModalDataset(Dataset):
    def __init__(
        self,
        processed_dir=Config.PROCESSED_DIR,
        clinical_dir=Config.PROCESSED_DIR / "clinical",
        transform=None,
        use_mask=True,
        csv_file=None,
        training=False,
        **kwargs,
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.use_mask = use_mask
        self.image_transform = Resize3D((128, 128, 128))
        self.mask_transform = MaskResize3D((128, 128, 128))

        if csv_file is not None and Path(csv_file).exists() and Path(csv_file).is_file():
            try:
                self.data = pd.read_csv(csv_file)
            except Exception:
                self.feature_df = pd.read_csv(Path(clinical_dir) / "clinical_features.csv")
                self.label_df = pd.read_csv(Path(clinical_dir) / "labels.csv")
                self.data = self.feature_df.merge(self.label_df, on="PatientID")
        else:
            self.feature_df = pd.read_csv(Path(clinical_dir) / "clinical_features.csv")
            self.label_df = pd.read_csv(Path(clinical_dir) / "labels.csv")
            self.data = self.feature_df.merge(self.label_df, on="PatientID")

        # =====================================================
        # Clinical Features
        # =====================================================
        ignore_columns = {
        "PatientID",
        "histology",
        "stage",
        "survival",
        "event",
}

        self.clinical_columns = [
            c for c in self.data.columns if c not in ignore_columns
        ]
        self.num_features = len(self.clinical_columns)

        self.patient_ids = []
        valid_rows = []

        for _, row in self.data.iterrows():
            pid = row["PatientID"]
            patient_folder = self.processed_dir / pid
            image_path = patient_folder / "image.npy"
            mask_path = patient_folder / "mask.npy"

            if not image_path.exists():
                continue

            if self.use_mask and not mask_path.exists():
                continue

            self.patient_ids.append(pid)
            valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        print("=" * 70)
        print("MultiModal Dataset")
        print("=" * 70)
        print("Patients :", len(self.data))
        print("Clinical Features :", self.num_features)
        print("=" * 70)

        self.training = training
        from datasets.transforms import RandomAugment3D
        self.augmentor = RandomAugment3D(training=self.training)

    def crop_roi(self, image, mask, margin_ratio=0.20):
        """
        Crops a tight 3D bounding box around the non-zero region of the mask with margin.
        """
        if mask is None or not (mask > 0.5).any():
            return image, mask

        nonzero = np.where(mask > 0.5)
        z_min, z_max = int(nonzero[0].min()), int(nonzero[0].max() + 1)
        y_min, y_max = int(nonzero[1].min()), int(nonzero[1].max() + 1)
        x_min, x_max = int(nonzero[2].min()), int(nonzero[2].max() + 1)

        z_margin = max(4, int(margin_ratio * (z_max - z_min)))
        y_margin = max(8, int(margin_ratio * (y_max - y_min)))
        x_margin = max(8, int(margin_ratio * (x_max - x_min)))

        z_min = max(0, z_min - z_margin)
        z_max = min(image.shape[0], z_max + z_margin)
        y_min = max(0, y_min - y_margin)
        y_max = min(image.shape[1], y_max + y_margin)
        x_min = max(0, x_min - x_margin)
        x_max = min(image.shape[2], x_max + x_margin)

        cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
        cropped_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        return cropped_image, cropped_mask

    def __len__(self):
        return len(self.data)

    def load_image(self, patient_id):
        image = np.load(self.processed_dir / patient_id / "image.npy").astype(np.float32)
        return image

    def load_mask(self, patient_id):
        mask = np.load(self.processed_dir / patient_id / "mask.npy").astype(np.float32)
        return mask

    def __getitem__(self, index):
        row = self.data.iloc[index]
        pid = row["PatientID"]

        # -----------------------------------------
        # Image & Mask Loading
        # -----------------------------------------
        image = self.load_image(pid)

        mask = None
        if self.use_mask:
            mask = self.load_mask(pid)
            image, mask = self.crop_roi(image, mask, margin_ratio=0.20)
            if mask is not None:
                image = image * mask

        # --------------------------------------------------
        # Resize Image & Mask
        # --------------------------------------------------
        image = self.image_transform(image)
        if mask is not None:
            mask = self.mask_transform(mask)

        # Tensor Conversion
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            elif not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=torch.float32)
            else:
                mask = mask.float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)

        # Apply 3D Augmentation during training
        if self.training:
            image, mask = self.augmentor(image, mask)

        # -----------------------------------------
        # Clinical Features
        # -----------------------------------------
        clinical = torch.tensor(
            row[self.clinical_columns].values.astype(np.float32)
        )

        # -----------------------------------------
        # Labels
        # -----------------------------------------
        histology = torch.tensor(int(row["histology"]), dtype=torch.long)
        stage = torch.tensor(int(row["stage"]), dtype=torch.long)
        survival = torch.tensor(float(row["survival"]), dtype=torch.float32)
        event = torch.tensor(float(row["event"]), dtype=torch.float32)

        sample = {
            "patient_id": pid,
            "image": image,
            "clinical": clinical,
            "histology": histology,
            "stage": stage,
            "survival": survival,
            "event": event,
        }

        if self.use_mask and mask is not None:
            sample["mask"] = mask.float()

        return sample


if __name__ == "__main__":

    dataset = MultiModalDataset()

    print()

    sample = dataset[0]

    print(sample["patient_id"])

    print(sample["image"].shape)

    print(sample["clinical"])

    print(sample["histology"])

    print(sample["stage"])

    print(sample["survival"])