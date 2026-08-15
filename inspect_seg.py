from preprocessing.seg_loader import SegLoader

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

loader = SegLoader.from_patient(ROOT, "LUNG1-100")

seg = loader.load()

print(type(seg))

print("\nAvailable Attributes:\n")

for attr in dir(seg):
    if not attr.startswith("_"):
        print(attr)