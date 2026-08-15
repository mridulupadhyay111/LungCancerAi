# test_qc.py

import SimpleITK as sitk

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.seg_loader import SegLoader
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.quality_control import QualityControl

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(ROOT)

image, volume, _ = reader.load_patient("LUNG1-100")

loader = SegLoader.from_patient(ROOT, "LUNG1-100")

mask = loader.load_binary_mask()

roi = ROIExtractor()

volume, mask, _ = roi.crop(volume, mask)

qc = QualityControl()

report = qc.validate(volume, mask)

print(report)