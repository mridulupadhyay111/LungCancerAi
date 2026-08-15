from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.hu_validation import HUValidator

DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(DATASET)

patient = reader.get_patients()[0]

image, volume, metadata = reader.load_patient(patient)

hu_volume, stats = HUValidator.validate(image)

print("=" * 60)

print("Patient :", patient.name)

print()

print("Shape :", hu_volume.shape)

print()

print(stats)

print("=" * 60)