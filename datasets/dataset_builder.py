"""
dataset_builder.py

Research-grade Dataset Builder for LungCancerAI

Pipeline
--------
Raw CT DICOM
        │
        ▼
Read CT
        ▼
Load Tumor Mask
        ▼
Resample
        ▼
ROI Crop
        ▼
HU Window
        ▼
Normalization
        ▼
Resize 128³
        ▼
Quality Control
        ▼
Save Dataset

Output
------

processed/

    images/
        LUNG1-001.npy

    labels/
        LUNG1-001.npy

    metadata/
        LUNG1-001.json

Author
------
LungCancerAI
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm

from configs.config import Config

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.seg_loader import SegLoader
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.windowing import CTWindowing
from preprocessing.normalization import (
    Normalizer,
    NormalizationType,
)
from preprocessing.resampling import Resampler
from preprocessing.quality_control import QualityControl
from preprocessing.image_utils import ImageUtils

from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Build Statistics
# ==========================================================

@dataclass
class BuildStatistics:

    discovered: int = 0

    processed: int = 0

    skipped: int = 0

    failed: int = 0

    warnings: int = 0

    failed_patients: List[str] = field(
        default_factory=list
    )


# ==========================================================
# Clinical Loader
# ==========================================================

class ClinicalDataLoader:
    """
    Loads the NSCLC clinical dataset.

    PatientID is used as the primary key.

    Returned format

    {
        patient_id:
        {
            age: ...
            gender: ...
            stage: ...
            ...
        }
    }
    """

    def __init__(self):

        self.directory = Path(Config.CLINICAL_DIR)

        self.data = {}

    # -----------------------------------------------------

    def discover_csv(self):

        csv_files = list(

            self.directory.glob("*.csv")

        )

        if len(csv_files) == 0:

            raise FileNotFoundError(

                f"No clinical csv inside\n"

                f"{self.directory}"

            )

        return csv_files[0]

    # -----------------------------------------------------

    def load(self):

        csv_file = self.discover_csv()

        logger.info(

            f"Loading Clinical CSV\n{csv_file}"

        )

        df = pd.read_csv(csv_file)

        columns = {

            c.lower(): c

            for c in df.columns

        }

        patient_column = None

        for key in [

            "patientid",

            "patient_id",

            "case",

            "subject",

            "id",

        ]:

            if key in columns:

                patient_column = columns[key]

                break

        if patient_column is None:

            raise RuntimeError(

                "Unable to identify Patient ID column."

            )

        for _, row in df.iterrows():

            patient = str(

                row[patient_column]

            ).strip()

            self.data[patient] = {

                col: (

                    row[col]

                    if not pd.isna(row[col])

                    else None

                )

                for col in df.columns

            }

        logger.info(

            f"Loaded {len(self.data)}"

            " clinical records."

        )

        return self.data

    # -----------------------------------------------------

    def get(self, patient):

        return self.data.get(

            patient,

            None,

        )


# ==========================================================
# Dataset Builder
# ==========================================================

class DatasetBuilder:

    def __init__(self):

        logger.info(

            "Initializing Dataset Builder..."

        )

        ##############################################

        self.reader = NSCLCDicomReader(

            Config.DICOM_DIR

        )

        self.roi = ROIExtractor()

        self.window = CTWindowing()

        self.normalizer = Normalizer(

            NormalizationType.Z_SCORE

        )

        self.resampler = Resampler(

            Config.TARGET_SPACING

        )

        self.qc = QualityControl()

        ##############################################

        self.clinical = ClinicalDataLoader()

        self.clinical.load()

        ##############################################

        self.stats = BuildStatistics()

        ##############################################

        self.output_root = Path(

            Config.PROCESSED_DIR

        )

        self.image_dir = (

            self.output_root / "images"

        )

        self.label_dir = (

            self.output_root / "labels"

        )

        self.metadata_dir = (

            self.output_root / "metadata"

        )

        ##############################################

        self.image_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.label_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.metadata_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        logger.info(

            "Output folders created."

        )

    # =====================================================
    # Discover Patients
    # =====================================================

    def discover_patients(self):

        patients = self.reader.get_patients()

        self.stats.discovered = len(

            patients

        )

        logger.info(

            f"Patients Found : "

            f"{len(patients)}"

        )

        return patients

    # =====================================================
    # Output Paths
    # =====================================================

    def image_path(

        self,

        patient,

    ):

        return (

            self.image_dir

            / f"{patient}.npy"

        )

    def label_path(

        self,

        patient,

    ):

        return (

            self.label_dir

            / f"{patient}.npy"

        )

    def metadata_path(

        self,

        patient,

    ):

        return (

            self.metadata_dir

            / f"{patient}.json"

        )

    # =====================================================
    # Already Processed?
    # =====================================================

    def is_processed(

        self,

        patient,

    ):

        return (

            self.image_path(patient).exists()

            and

            self.label_path(patient).exists()

            and

            self.metadata_path(patient).exists()

        )

    # =====================================================
    # Remove Corrupted Output
    # =====================================================

    def remove_existing(

        self,

        patient,

    ):

        for file in [

            self.image_path(patient),

            self.label_path(patient),

            self.metadata_path(patient),

        ]:

            if file.exists():

                file.unlink()

    # =====================================================
    # Save Metadata
    # =====================================================

    def save_metadata(

        self,

        patient,

        metadata,

    ):

        with open(

            self.metadata_path(patient),

            "w",

        ) as f:

            json.dump(

                metadata,

                f,

                indent=4,

            )

    # =====================================================
    # Resize Volume
    # =====================================================

    def resize_volume(

        self,

        volume,

    ):

        """
        Part 2 will implement this using
        SimpleITK.
        """

        raise NotImplementedError
        # =====================================================
    # Resize Volume
    # =====================================================

    def resize_volume(
        self,
        volume: np.ndarray,
        is_mask: bool = False,
    ) -> np.ndarray:
        """
        Resize a 3D volume to Config.TARGET_SIZE.
        Linear interpolation for CT.
        Nearest neighbour for masks.
        """

        image = sitk.GetImageFromArray(volume)

        original_size = image.GetSize()
        target_size = Config.TARGET_SIZE

        original_spacing = image.GetSpacing()

        new_spacing = [
            original_spacing[i] * original_size[i] / target_size[i]
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(
          [int(x) for x in target_size[::-1]]
)

        resampler.SetOutputSpacing(new_spacing)

        resampler.SetOutputDirection(
            image.GetDirection()
        )

        resampler.SetOutputOrigin(
            image.GetOrigin()
        )

        if is_mask:

            resampler.SetInterpolator(
                sitk.sitkNearestNeighbor
            )

        else:

            resampler.SetInterpolator(
                sitk.sitkLinear
            )

        resized = resampler.Execute(image)

        return sitk.GetArrayFromImage(resized)

    # =====================================================
    # Process One Patient
    # =====================================================

    def process_patient(
        self,
        patient_id: str,
    ):

        logger.info(
            f"Processing {patient_id}"
        )

        ####################################################
        # Load CT
        ####################################################

        ct_image, _, meta = self.reader.load_patient(
            patient_id
        )

        ####################################################
        # Load Tumor Segmentation
        ####################################################

        seg_loader = SegLoader.from_patient(
            Config.DICOM_DIR,
            patient_id,
        )

        mask = seg_loader.load_binary_mask()

        ####################################################
        # Convert mask to SimpleITK
        ####################################################

        mask_image = ImageUtils.numpy_to_sitk(
            mask.astype(np.uint8),
            ct_image,
        )

        ####################################################
        # Geometry Check
        ####################################################

        ImageUtils.validate_geometry(
            ct_image,
            mask_image,
        )

        ####################################################
        # Resample
        ####################################################

        ct_image, _ = self.resampler.resample_ct(
            ct_image
        )

        mask_image, _ = self.resampler.resample_mask(
            mask_image
        )

        ####################################################
        # Convert to NumPy
        ####################################################

        image = ImageUtils.sitk_to_numpy(
            ct_image
        ).astype(np.float32)

        mask = ImageUtils.sitk_to_numpy(
            mask_image
        ).astype(np.uint8)

        ####################################################
        # Crop Tumor ROI
        ####################################################

        image, mask, bbox = self.roi.crop(
            image,
            mask,
        )

        ####################################################
        # Windowing
        ####################################################

        image = self.window.apply(
            image
        )

        ####################################################
        # Normalization
        ####################################################

        image = self.normalizer.apply(
            image
        )

        ####################################################
        # Resize to 128³
        ####################################################

        image = self.resize_volume(
            image,
            is_mask=False,
        )

        mask = self.resize_volume(
            mask,
            is_mask=True,
        )

        ####################################################
        # Force dtypes
        ####################################################

        image = image.astype(np.float32)

        mask = mask.astype(np.uint8)

        ####################################################
        # QC
        ####################################################

        report = self.qc.validate(
            image,
            mask,
        )

        if not report.passed:

            raise RuntimeError(
                "\n".join(report.errors)
            )

        ####################################################
        # Clinical Record
        ####################################################

        clinical = self.clinical.get(
            patient_id
        )

        if clinical is None:

            logger.warning(
                f"No clinical data for {patient_id}"
            )

            clinical = {}

        ####################################################
        # Sanity Checks
        ####################################################

        assert image.shape == Config.TARGET_SIZE

        assert mask.shape == Config.TARGET_SIZE

        assert image.dtype == np.float32

        assert mask.dtype == np.uint8

        ####################################################
        # Save Image
        ####################################################

        np.save(
            self.image_path(patient_id),
            image,
        )

        ####################################################
        # Save Mask
        ####################################################

        np.save(
            self.label_path(patient_id),
            mask,
        )

        ####################################################
        # Metadata
        ####################################################
        metadata = {

       "patient_id": str(patient_id),

       "original_shape": [int(x) for x in meta["shape"]],

       "processed_shape": [int(x) for x in ct_np.shape],

       "bbox": {

        "z_min": int(bbox.z_min),
        "z_max": int(bbox.z_max),

        "y_min": int(bbox.y_min),
        "y_max": int(bbox.y_max),

        "x_min": int(bbox.x_min),
        "x_max": int(bbox.x_max),

    },

      "qc_passed": bool(report.passed),

       "warnings": [str(w) for w in report.warnings]

}

        self.save_metadata(
            patient_id,
            metadata,
        )

        logger.info(
            f"Saved {patient_id}"
        )
        # =====================================================
    # Build Dataset
    # =====================================================

    def build(self):

        logger.info("=" * 80)
        logger.info("Starting Dataset Build")
        logger.info("=" * 80)

        patients = self.discover_patients()

        progress = tqdm(
            patients,
            total=len(patients),
            desc="Building Dataset",
        )

        for patient_id in progress:

            progress.set_postfix(
                patient=patient_id
            )

            try:

                if self.is_processed(patient_id):

                    logger.info(
                        f"{patient_id} already processed."
                    )

                    self.stats.skipped += 1

                    continue

                self.process_patient(patient_id)

                self.stats.processed += 1

            except KeyboardInterrupt:

                logger.warning(
                    "Dataset build interrupted."
                )

                raise

            except Exception as e:

                logger.error(
                    f"Failed: {patient_id}"
                )

                logger.error(str(e))

                logger.debug(
                    traceback.format_exc()
                )

                self.remove_existing(patient_id)

                self.stats.failed += 1

                self.stats.failed_patients.append(
                    patient_id
                )

        self.print_summary()

    # =====================================================
    # Summary
    # =====================================================

    def print_summary(self):

        logger.info("")
        logger.info("=" * 80)
        logger.info("DATASET BUILD SUMMARY")
        logger.info("=" * 80)

        logger.info(
            f"Patients Found      : {self.stats.discovered}"
        )

        logger.info(
            f"Successfully Built  : {self.stats.processed}"
        )

        logger.info(
            f"Skipped             : {self.stats.skipped}"
        )

        logger.info(
            f"Failed              : {self.stats.failed}"
        )

        logger.info(
            f"Warnings            : {self.stats.warnings}"
        )

        if self.stats.failed_patients:

            logger.info("")
            logger.info("Failed Patients")

            for patient in self.stats.failed_patients:

                logger.info(f"  - {patient}")

        logger.info("=" * 80)

    # =====================================================
    # Export Build Report
    # =====================================================

    def export_report(self):

        report = {

            "patients_found":
                self.stats.discovered,

            "processed":
                self.stats.processed,

            "skipped":
                self.stats.skipped,

            "failed":
                self.stats.failed,

            "warnings":
                self.stats.warnings,

            "failed_patients":
                self.stats.failed_patients,

        }

        report_file = (
            self.output_root /
            "build_report.json"
        )

        with open(
            report_file,
            "w",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
            )

        logger.info(
            f"Report saved to {report_file}"
        )

    # =====================================================
    # Verify Dataset
    # =====================================================

    def verify(self):

        logger.info("")
        logger.info("=" * 80)
        logger.info("VERIFYING DATASET")
        logger.info("=" * 80)

        images = list(
            self.image_dir.glob("*.npy")
        )

        masks = list(
            self.label_dir.glob("*.npy")
        )

        metadata = list(
            self.metadata_dir.glob("*.json")
        )

        logger.info(
            f"Images   : {len(images)}"
        )

        logger.info(
            f"Masks    : {len(masks)}"
        )

        logger.info(
            f"Metadata : {len(metadata)}"
        )

        if (
            len(images)
            != len(masks)
            or
            len(images)
            != len(metadata)
        ):

            raise RuntimeError(
                "Dataset verification failed."
            )

        logger.info(
            "Verification Passed."
        )

    # =====================================================
    # Statistics
    # =====================================================

    def statistics(self):

        return {

            "patients":
                self.stats.discovered,

            "processed":
                self.stats.processed,

            "skipped":
                self.stats.skipped,

            "failed":
                self.stats.failed,

        }


# ==========================================================
# Factory
# ==========================================================

def build_dataset():

    builder = DatasetBuilder()

    builder.build()

    builder.verify()

    builder.export_report()

    return builder


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    builder = build_dataset()

    print()
    print("=" * 80)
    print("Dataset Build Complete")
    print("=" * 80)

    print(builder.statistics())        