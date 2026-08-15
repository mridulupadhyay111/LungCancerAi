from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.resample import Resampler

DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(DATASET)

patient = reader.get_patients()[0]

image, _, _ = reader.load_patient(patient)

print("=" * 60)
print("ORIGINAL")
print("Spacing :", image.GetSpacing())
print("Size    :", image.GetSize())

resampler = Resampler()

new_image, info = resampler.resample_ct(image)

print("\nRESAMPLED")
print("Spacing :", new_image.GetSpacing())
print("Size    :", new_image.GetSize())

Resampler.validate(new_image)

print("\nValidation : PASSED")
print("=" * 60)