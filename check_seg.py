from pathlib import Path
import pydicom

seg_file = Path(
    r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics\LUNG1-100"
)

# Find the SEG file automatically
study = next(seg_file.iterdir())
seg_folder = next(p for p in study.iterdir() if p.name.startswith("SEG"))
dcm = next(seg_folder.glob("*.dcm"))

print("SEG File:", dcm)

ds = pydicom.dcmread(dcm)

print("\nSOP Class UID:")
print(ds.SOPClassUID)

print("\nModality:")
print(ds.Modality)

print("\nManufacturer:")
print(getattr(ds, "Manufacturer", "Unknown"))

print("\nRows:", ds.Rows)
print("Columns:", ds.Columns)
print("Number of Frames:", getattr(ds, "NumberOfFrames", 1))