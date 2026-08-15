import SimpleITK as sitk

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.windowing import CTWindowing
from preprocessing.normalization import (
    Normalizer,
    NormalizationType
)

ROOT = r"D:\LungCancerAI\data\raw\dicom\nsclc_radiomics"

reader = NSCLCDicomReader(ROOT)

image, volume, metadata = reader.load_patient("LUNG1-100")

window = CTWindowing()

volume = window.apply(volume)

normalizer = Normalizer(
    NormalizationType.Z_SCORE
)

volume = normalizer.apply(volume)

normalizer.summary(volume)