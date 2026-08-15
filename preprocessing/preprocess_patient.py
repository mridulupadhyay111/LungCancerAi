from pathlib import Path
import json

import numpy as np
import SimpleITK as sitk
from PIL import Image

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.seg_loader import SegLoader
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.resample import Resampler
from preprocessing.windowing import CTWindowing


class PatientPreprocessor:
    """
    Research-grade preprocessing pipeline for one patient.
    """

    def __init__(self, dataset_root, output_root):

        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

        self.reader = NSCLCDicomReader(dataset_root)

        self.roi = ROIExtractor(margin=20)

        self.resampler = Resampler()

        self.window =  CTWindowing()

    def process(self, patient_id):

        print("=" * 60)
        print(f"Processing Patient : {patient_id}")
        print("=" * 60)

        # -------------------------------
        # Load CT
        # -------------------------------
        image, _, _ = self.reader.load_patient(patient_id)

        ct = sitk.GetArrayFromImage(image)

        print("Original CT Shape :", ct.shape)

        # -------------------------------
        # Load Tumor Mask
        # -------------------------------
        loader = SegLoader.from_patient(
            self.dataset_root,
            patient_id
        )

        mask = loader.load_binary_mask()

        print("Mask Shape :", mask.shape)

        # -------------------------------
        # Crop ROI
        # -------------------------------
        ct, mask, bbox = self.roi.crop(ct, mask)

        print("ROI Shape :", ct.shape)

        # -------------------------------
        # Windowing
        # -------------------------------
        ct = self.window.apply(ct)

        # -------------------------------
        # Create Output Folder
        # -------------------------------
        patient_dir = self.output_root / patient_id

        patient_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------------
        # Save Arrays
        # -------------------------------
        np.save(patient_dir / "ct.npy", ct)

        np.save(patient_dir / "mask.npy", mask)

        # -------------------------------
        # Metadata
        # -------------------------------
        metadata = {

            "patient_id": patient_id,

            "ct_shape": list(ct.shape),

            "mask_shape": list(mask.shape),

            "tumor_voxels": int(mask.sum()),

            "bounding_box": {
                "z_min": int(bbox.z_min),
                "z_max": int(bbox.z_max),
                "y_min": int(bbox.y_min),
                "y_max": int(bbox.y_max),
                "x_min": int(bbox.x_min),
                "x_max": int(bbox.x_max)
            }

        }

        with open(
            patient_dir / "metadata.json",
            "w"
        ) as f:

            json.dump(metadata, f, indent=4)

        # -------------------------------
        # Preview Image
        # -------------------------------
        middle = ct[ct.shape[0] // 2]

        middle = (
            (middle - middle.min())
            /
            (middle.max() - middle.min() + 1e-8)
            * 255
        ).astype(np.uint8)

        Image.fromarray(middle).save(
            patient_dir / "preview.png"
        )

        print("\nPreprocessing Completed Successfully.")
        print("Saved To :", patient_dir)
        print("=" * 60)