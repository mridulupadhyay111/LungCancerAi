import SimpleITK as sitk

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.seg_loader import SegLoader
from preprocessing.roi_extractor import ROIExtractor

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(ROOT)

patient = reader.get_patients()[0]

image, _, _ = reader.load_patient(patient)

image = sitk.GetArrayFromImage(image)

loader = SegLoader.from_patient(ROOT, patient)

mask = loader.load_binary_mask()

extractor = ROIExtractor(margin=20)

cropped_image, cropped_mask, bbox = extractor.crop(
    image,
    mask
)

print("=" * 60)

print("Original :", image.shape)
print("ROI      :", cropped_image.shape)
print("Mask Sum :", cropped_mask.sum())
print(bbox)

print("=" * 60)