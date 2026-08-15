from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.resample import Resampler

DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(DATASET)

patient = reader.get_patients()[0]

image, _, _ = reader.load_patient(patient)

print("=" * 60)
print("Original")
print("Spacing :", image.GetSpacing())
print("Size    :", image.GetSize())

resampler = Resampler(target_spacing=(1.0, 1.0, 1.0))

new_image, meta = resampler.resample(image)

print("\nResampled")
print("Spacing :", new_image.GetSpacing())
print("Size    :", new_image.GetSize())

print("=" * 60)