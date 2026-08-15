"""
config.py

Central configuration for preprocessing pipeline.
"""

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:

    # ROI
    roi_margin: int = 20

    # Target voxel spacing (mm)
    target_spacing = (1.0, 1.0, 1.0)

    # Lung window
    window_level = -600
    window_width = 1500

    # Normalization
    normalization = "zscore"

    # Minimum tumor voxels
    minimum_tumor_voxels = 100

    # Output dtype
    image_dtype = "float32"

    mask_dtype = "uint8"

    # Save preview
    save_preview = True

    # Compression
    compression = True