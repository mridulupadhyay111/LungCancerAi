from preprocessing.dicom_reader import NSCLCDicomReader

reader = NSCLCDicomReader(
    r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"
)

patients = reader.get_patients()

print(f"Patients Found : {len(patients)}")

print(patients[:5])

image, volume, metadata = reader.load_patient(
    patients[0]
)

print(metadata)