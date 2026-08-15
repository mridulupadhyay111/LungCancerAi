"""
volume_builder.py

Build a 3D CT volume from a DICOM series using SimpleITK.

Author: Mridul Upadhyay
Project: LungCancerAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

from utils.logger import get_logger

logger = get_logger()


class VolumeBuilder:
    """
    Reads a DICOM series using SimpleITK and converts it
    into a NumPy volume while preserving metadata.
    """

    def __init__(self, dicom_folder: str | Path):

        self.dicom_folder = Path(dicom_folder)

        if not self.dicom_folder.exists():
            raise FileNotFoundError(
                f"DICOM folder does not exist: {self.dicom_folder}"
            )

    def _get_series_files(self):
        """
        Retrieve all DICOM filenames belonging to the series.
        """

        reader = sitk.ImageSeriesReader()

        series_ids = reader.GetGDCMSeriesIDs(str(self.dicom_folder))

        if not series_ids:
            raise RuntimeError(
                f"No DICOM series found inside {self.dicom_folder}"
            )

        # Use first series by default
        series_uid = series_ids[0]

        file_names = reader.GetGDCMSeriesFileNames(
            str(self.dicom_folder),
            series_uid
        )

        return file_names

    def build(self) -> Tuple[np.ndarray, sitk.Image, Dict]:
        """
        Returns
        -------
        volume : np.ndarray
            Shape = (Slices, Height, Width)

        image : SimpleITK.Image

        metadata : dict
        """

        logger.info("Reading DICOM series...")

        file_names = self._get_series_files()

        reader = sitk.ImageSeriesReader()

        reader.SetFileNames(file_names)

        image = reader.Execute()

        volume = sitk.GetArrayFromImage(image).astype(np.int16)

        metadata = {

            "shape": volume.shape,

            "spacing": image.GetSpacing(),

            "origin": image.GetOrigin(),

            "direction": image.GetDirection(),

            "pixel_type": image.GetPixelIDTypeAsString(),

            "num_slices": volume.shape[0]
        }

        logger.info(
            f"Volume Shape : {volume.shape}"
        )

        logger.info(
            f"Voxel Spacing : {metadata['spacing']}"
        )

        return volume, image, metadata