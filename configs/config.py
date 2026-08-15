"""
Global configuration for LungCancerAI.

All preprocessing, training, evaluation and deployment
modules should import values from this file instead of
hardcoding parameters.
"""

from pathlib import Path


class Config:

    # =====================================================
    # PROJECT
    # =====================================================

    PROJECT_ROOT = Path(r"D:\LungCancerAI")

    # =====================================================
    # DATA
    # =====================================================

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

    DICOM_DIR = RAW_DATA_DIR / "dicom" / "nsclc_radiomics"

    CLINICAL_DIR = RAW_DATA_DIR / "clinical"

    SEGMENTATION_DIR = RAW_DATA_DIR / "segmentation"

    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    # =====================================================
    # OUTPUT FOLDERS
    # =====================================================

    HU_DIR = INTERIM_DIR / "hu"

    WINDOW_DIR = INTERIM_DIR / "windowed"

    RESAMPLE_DIR = INTERIM_DIR / "resampled"

    MASK_DIR = INTERIM_DIR / "lung_masks"

    CROPPED_DIR = INTERIM_DIR / "cropped"

    NORMALIZED_DIR = PROCESSED_DIR / "images"

    LABEL_DIR = PROCESSED_DIR / "labels"

    METADATA_DIR = PROCESSED_DIR / "metadata"

    # =====================================================
    # CT PARAMETERS
    # =====================================================

    TARGET_SPACING = (1.0, 1.0, 1.0)

    WINDOW_LEVEL = -600

    WINDOW_WIDTH = 1500

    TARGET_SIZE = (128, 128, 128)

    # =====================================================
    # NORMALIZATION
    # =====================================================

    MIN_HU = -1000

    MAX_HU = 400

    # =====================================================
    # RANDOMNESS
    # =====================================================

    RANDOM_SEED = 42

    NUM_WORKERS = 8

    # =====================================================
    # LOGGING
    # =====================================================

    LOG_LEVEL = "INFO"