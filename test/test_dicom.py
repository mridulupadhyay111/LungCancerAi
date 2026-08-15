from preprocessing.dicom_reader import NSCLCDicomReader

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(ROOT)

reader.summary()

image, volume, metadata = reader.load_patient("LUNG1-100")

print(volume.shape)
print(metadata)