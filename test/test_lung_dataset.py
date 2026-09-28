import json
from pathlib import Path

import numpy as np

from datasets.lung_dataset import LungDataset


def test_lung_dataset_ignores_non_patient_dirs(tmp_path):
    patient_dir = tmp_path / "LUNG1-001"
    patient_dir.mkdir()
    np.save(patient_dir / "image.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.save(patient_dir / "mask.npy", np.zeros((2, 2, 2), dtype=np.uint8))
    (patient_dir / "metadata.json").write_text(json.dumps({"id": "LUNG1-001"}))

    unrelated_dir = tmp_path / "clinical"
    unrelated_dir.mkdir()

    dataset = LungDataset(root_dir=tmp_path, use_masks=True, return_metadata=True, cache=False)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["patient_id"] == "LUNG1-001"
    assert sample["image"].shape[0] == 1
    assert sample["mask"].shape[0] == 1
    assert sample["metadata"]["id"] == "LUNG1-001"
