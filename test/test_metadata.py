from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.metadata import MetadataExtractor

reader = NSCLCDicomReader(
    r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"
)

patient = reader.get_patients()[0]

ct_folder = reader.find_ct_folder(patient)

metadata = MetadataExtractor.extract(ct_folder)

for k, v in metadata.items():
    print(f"{k:25} : {v}")