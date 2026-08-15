import numpy as np
from pathlib import Path
from preprocessing.seg_loader import SegLoader

ROOT = Path(r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics")

patient = "LUNG1-105"

study = next((ROOT / patient).iterdir())

seg_folder = next(
    p for p in study.iterdir()
    if p.name.startswith("SEG")
)

seg_file = next(seg_folder.glob("*.dcm"))

loader = SegLoader(seg_file)

mask = loader.load_binary_mask()

print("=" * 60)
print("Tumor Mask")
print("=" * 60)
print("Shape :", mask.shape)
print("Data type :", mask.dtype)
print("Unique values :", np.unique(mask))
