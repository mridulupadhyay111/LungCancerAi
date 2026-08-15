"""
lung_dataset.py

Research-grade Multimodal Dataset
for LungCancerAI

Loads
------
✓ CT Volume
✓ Tumor Mask
✓ Clinical Features
✓ Histology Label
✓ Stage Label
✓ Survival Label
✓ Metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transforms.ct_transforms import MaskResize3D, Resize3D

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import Config


class LungDataset(Dataset):

    def __init__(
        self,
        root_dir=Config.PROCESSED_DIR,
        transform: Optional[Callable] = None,
        use_masks=True,
        return_metadata=True,
        cache=False,
    ):

        self.image_transform = Resize3D((128, 128, 128))
        self.mask_transform = MaskResize3D((128, 128, 128))

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.use_masks = use_masks

        self.return_metadata = return_metadata

        self.cache = cache

        if not self.root_dir.exists():

            raise FileNotFoundError(self.root_dir)

        # ---------------------------------------
        # Clinical tables
        # ---------------------------------------

        clinical_dir = Config.PROCESSED_DIR / "clinical"

        self.features = pd.read_csv(
            clinical_dir / "clinical_features.csv"
        )

        self.labels = pd.read_csv(
            clinical_dir / "labels.csv"
        )

        self.features.set_index(
            "PatientID",
            inplace=True
        )

        self.labels.set_index(
            "PatientID",
            inplace=True
        )

        # ---------------------------------------
        # Patient folders
        # ---------------------------------------

        self.patient_dirs = sorted(

            [

                p

                for p in self.root_dir.iterdir()

                if p.is_dir()

                and p.name.startswith("LUNG1")

            ]

        )

        if len(self.patient_dirs) == 0:

            raise RuntimeError(

                "No processed patients found."

            )

        self.memory_cache = {}

        print("=" * 60)

        print("LungDataset Initialized")

        print("=" * 60)

        print("Patients :", len(self.patient_dirs))

        print("Clinical :", len(self.features))

        print("=" * 60)

    # ------------------------------------------

    def __len__(self):

        return len(self.patient_dirs)

    # ------------------------------------------

    def load_numpy(self, path):

        if not path.exists():

            raise FileNotFoundError(path)

        return np.load(path)

    # ------------------------------------------

    def load_metadata(self, path):

     print(f"\nLoading metadata -> {path}")

     with open(path, "r") as f:

        try:

            return json.load(f)

        except Exception as e:

            print("\nERROR IN FILE:")

            print(path)

            raise e
    # ------------------------------------------

    def get_patient(self, index):

        patient_dir = self.patient_dirs[index]

        patient_id = patient_dir.name

        image = self.load_numpy(

            patient_dir / "image.npy"

        ).astype(np.float32)

        mask = self.load_numpy(

            patient_dir / "mask.npy"

        ).astype(np.uint8)

        metadata = self.load_metadata(

            patient_dir / "metadata.json"

        )

        return patient_id, image, mask, metadata
        # -----------------------------------------------------

    def __getitem__(self, index):

        # -----------------------------
        # Cache
        # -----------------------------

        if self.cache and index in self.memory_cache:

            return self.memory_cache[index]

        # -----------------------------
        # Load patient files
        # -----------------------------

        patient_id, image, mask, metadata = self.get_patient(index)

        # -----------------------------
        # Load Clinical Features
        # -----------------------------

        if patient_id not in self.features.index:

            raise RuntimeError(

                f"{patient_id} not found in clinical_features.csv"

            )

        if patient_id not in self.labels.index:

            raise RuntimeError(

                f"{patient_id} not found in labels.csv"

            )

        feature_row = self.features.loc[patient_id]

        label_row = self.labels.loc[patient_id]

        # -----------------------------
        # Clinical Tensor
        # -----------------------------

        clinical = torch.tensor(

            [

                float(feature_row["age"]),

                float(feature_row["gender"]),

                float(feature_row["T_stage"]),

                float(feature_row["N_stage"]),

                float(feature_row["M_stage"])

            ],

            dtype=torch.float32,

        )

        # -----------------------------
        # Labels
        # -----------------------------

        histology = torch.tensor(

            int(label_row["histology"]),

            dtype=torch.long,

        )

        stage = torch.tensor(

            int(label_row["stage"]),

            dtype=torch.long,

        )

        survival = torch.tensor(

            float(label_row["survival"]),

            dtype=torch.float32,

        )

        event = torch.tensor(

            float(label_row["event"]),

            dtype=torch.float32,

        )

        # -----------------------------
        # Image
        # -----------------------------

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        # -----------------------------
        # Mask
        # -----------------------------

        if self.use_masks:

            mask = self.mask_transform(mask)

            mask = mask.long()

            if mask.ndim == 3:

                mask = mask.unsqueeze(0)

        # -----------------------------
        # Sample Dictionary
        # -----------------------------

        sample = {

            "patient_id": patient_id,

            "image": image,

            "clinical": clinical,

            "histology": histology,

            "stage": stage,

            "survival": survival,

            "event": event,

        }

        if self.use_masks:

            sample["mask"] = mask

        if self.return_metadata:

            sample["metadata"] = metadata

        if self.cache:

            self.memory_cache[index] = sample

        return sample
        # -----------------------------------------------------
    # Return all Patient IDs
    # -----------------------------------------------------

    def get_patient_ids(self):

        return [p.name for p in self.patient_dirs]

    # -----------------------------------------------------
    # Check whether patient exists
    # -----------------------------------------------------

    def contains(self, patient_id):

        return patient_id in self.get_patient_ids()

    # -----------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------

    def summary(self):

        print("=" * 70)
        print("LungCancerAI Dataset Summary")
        print("=" * 70)

        print(f"Dataset Root      : {self.root_dir}")
        print(f"Patients          : {len(self)}")
        print(f"Clinical Records  : {len(self.features)}")
        print(f"Use Masks         : {self.use_masks}")
        print(f"Return Metadata   : {self.return_metadata}")
        print(f"Cache             : {self.cache}")

        print("=" * 70)

    # -----------------------------------------------------
    # Verify Dataset
    # -----------------------------------------------------

    def verify(self):

        print()
        print("=" * 70)
        print("Verifying Dataset")
        print("=" * 70)

        missing_clinical = 0
        missing_labels = 0
        missing_files = 0

        for patient_dir in self.patient_dirs:

            pid = patient_dir.name

            if pid not in self.features.index:
                print(f"Missing Clinical : {pid}")
                missing_clinical += 1

            if pid not in self.labels.index:
                print(f"Missing Label : {pid}")
                missing_labels += 1

            if not (patient_dir / "image.npy").exists():
                print(f"Missing image : {pid}")
                missing_files += 1

            if self.use_masks:

                if not (patient_dir / "mask.npy").exists():
                    print(f"Missing mask : {pid}")
                    missing_files += 1

        print()

        print(f"Missing Clinical : {missing_clinical}")
        print(f"Missing Labels   : {missing_labels}")
        print(f"Missing Files    : {missing_files}")

        if (
            missing_clinical == 0
            and
            missing_labels == 0
            and
            missing_files == 0
        ):

            print()
            print("✓ Dataset Verification Passed")

        else:

            print()
            print("Dataset contains problems.")

        print("=" * 70)

    # -----------------------------------------------------
    # Print One Sample
    # -----------------------------------------------------

    def inspect(self, index=0):

        sample = self[index]

        print("=" * 70)
        print("Sample Inspection")
        print("=" * 70)

        print("Patient ID :", sample["patient_id"])

        print("Image Shape :", sample["image"].shape)

        if "mask" in sample:

            print("Mask Shape :", sample["mask"].shape)

        print("Clinical :", sample["clinical"])

        print("Histology :", sample["histology"])

        print("Stage :", sample["stage"])

        print("Survival :", sample["survival"])

        print("=" * 70)


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    dataset = LungDataset(
        root_dir=Config.PROCESSED_DIR,
        use_masks=True,
        return_metadata=True,
        cache=False,
    )

    dataset.summary()

    dataset.verify()

    dataset.inspect(0)

    print()

    print("First 5 Patients")

    for pid in dataset.get_patient_ids()[:5]:

        print(pid)

    print()

    sample = dataset[0]

    print("Dictionary Keys")

    print(sample.keys())

    print()

    print("Image Shape")

    print(sample["image"].shape)

    print()

    print("Clinical Shape")

    print(sample["clinical"].shape)

    print()

    print("Histology")

    print(sample["histology"])

    print()

    print("Stage")

    print(sample["stage"])

    print()

    print("Survival")

    print(sample["survival"])

    print()

    print("=" * 70)
    print("Dataset Ready For Training")
    print("=" * 70)        