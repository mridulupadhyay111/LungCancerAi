from preprocessing.seg_loader import SegLoader

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

loader = SegLoader.from_patient(
    ROOT,
    "LUNG1-100"
)

loader.summary()