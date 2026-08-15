from preprocessing.preprocess_patient import PatientPreprocessor

DATASET = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

OUTPUT = r"D:\LungCancerAI\data\processed"

processor = PatientPreprocessor(
    DATASET,
    OUTPUT
)

processor.process("LUNG1-100")