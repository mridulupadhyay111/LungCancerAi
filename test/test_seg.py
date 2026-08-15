from pathlib import Path

from preprocessing.seg_loader import SegLoader

ROOT = Path(r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics")

patient = "LUNG1-100"

study = next((ROOT / patient).iterdir())

seg_folder = next(
    f for f in study.iterdir()
    if f.name.startswith("SEG")
)

seg_file = next(seg_folder.glob("*.dcm"))

loader = SegLoader(seg_file)

loader.summary()