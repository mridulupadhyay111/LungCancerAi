from pathlib import Path

from preprocessing.ct_loader import CTLoader

ROOT = Path(
    r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"
)

patient = "LUNG1-100"

study = next((ROOT / patient).iterdir())

ct_folder = next(
    p for p in study.iterdir()
    if p.name.startswith("CT")
)

loader = CTLoader(ct_folder)

loader.summary()