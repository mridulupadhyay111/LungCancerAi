from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.hu_validation import HUValidator
from preprocessing.windowing import CTWindowing

# Dataset path
DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

# Initialize reader
reader = NSCLCDicomReader(DATASET)

# Load first patient
patient = reader.get_patients()[0]

image, volume, metadata = reader.load_patient(patient)

# Validate HU (this returns hu_volume)
hu_volume, hu_stats = HUValidator.validate(image)

# Apply lung window
window = CTWindowing()

windowed_volume = window.apply(hu_volume)

print("=" * 60)
print(f"Patient : {patient.name}")
print(f"Shape   : {windowed_volume.shape}")
print(f"Min     : {windowed_volume.min():.4f}")
print(f"Max     : {windowed_volume.max():.4f}")
print(f"Mean    : {windowed_volume.mean():.4f}")
print(f"Window  : {window.get_parameters()}")
print("=" * 60)