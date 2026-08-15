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
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.use_mask = use_mask
        self.image_transform = Resize3D((128, 128, 128))
        self.mask_transform = MaskResize3D((128, 128, 128))

        self.feature_df = pd.read_csv(clinical_dir / "clinical_features.csv")
        self.label_df = pd.read_csv(clinical_dir / "labels.csv")
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
        # Image
        # -----------------------------------------
        image = self.load_image(pid)

        # -----------------------------------------
        # Mask
        # -----------------------------------------
        mask = None
        if self.use_mask:
            mask = self.load_mask(pid)
            image = image * mask

       # --------------------------------------------------
# Resize Image
# --------------------------------------------------

        image = self.image_transform(image)

# --------------------------------------------------
# Resize Mask
# --------------------------------------------------

        if mask is not None:

         mask = self.mask_transform(mask)
        # Tensor Conversion
        # -----------------------------------------
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

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
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
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