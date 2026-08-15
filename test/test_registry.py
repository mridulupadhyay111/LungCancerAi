from preprocessing.patient_registry import PatientRegistry

DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

OUTPUT = r"D:\LungCancerAI\data\processed\metadata"

registry = PatientRegistry(DATASET)

df = registry.build()

print(df.head())

registry.save(df, OUTPUT)