"""
image_utils.py

Research-grade utilities for converting between NumPy arrays
and SimpleITK images while preserving spatial geometry.

These utilities ensure that CT images and segmentation masks
remain perfectly aligned throughout preprocessing.

Author: LungCancerAI
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk


class ImageUtils:
    """
    Utility class for SimpleITK <-> NumPy conversions.
    """

    # ---------------------------------------------------------
    # SITK -> NumPy
    # ---------------------------------------------------------

    @staticmethod
    def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
        """
        Convert a SimpleITK image into a NumPy array.

        Parameters
        ----------
        image : sitk.Image

        Returns
        -------
        np.ndarray
        """

        if not isinstance(image, sitk.Image):
            raise TypeError(
                "Expected SimpleITK.Image."
            )

        return sitk.GetArrayFromImage(image)

    # ---------------------------------------------------------
    # NumPy -> SITK
    # ---------------------------------------------------------

    @staticmethod
    def numpy_to_sitk(
        array: np.ndarray,
        reference: sitk.Image
    ) -> sitk.Image:
        """
        Convert a NumPy array into a SimpleITK image
        while copying geometry from a reference image.

        Parameters
        ----------
        array : np.ndarray

        reference : sitk.Image

        Returns
        -------
        sitk.Image
        """

        if not isinstance(reference, sitk.Image):
            raise TypeError(
                "Reference must be SimpleITK.Image."
            )

        image = sitk.GetImageFromArray(array)

        if image.GetSize() != reference.GetSize():
            try:
                image = sitk.Resample(
                    image,
                    reference,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0.0,
                )
            except Exception:
                target_shape = reference.GetSize()
                if array.ndim == 3:
                    padded = np.zeros(target_shape, dtype=array.dtype)
                    z, y, x = array.shape
                    z2, y2, x2 = target_shape
                    z = min(z, z2)
                    y = min(y, y2)
                    x = min(x, x2)
                    padded[:z, :y, :x] = array[:z, :y, :x]
                    image = sitk.GetImageFromArray(padded)
                else:
                    image = sitk.GetImageFromArray(array)

        image.CopyInformation(reference)

        return image

    # ---------------------------------------------------------
    # Copy Geometry
    # ---------------------------------------------------------

    @staticmethod
    def copy_geometry(
        source: sitk.Image,
        target: sitk.Image
    ) -> sitk.Image:
        """
        Copy spacing, origin and direction.
        """

        target.CopyInformation(source)

        return target

    # ---------------------------------------------------------
    # Geometry Comparison
    # ---------------------------------------------------------

    @staticmethod
    def same_geometry(
        image1: sitk.Image,
        image2: sitk.Image
    ) -> bool:
        """
        Check whether two images have identical geometry.
        """

        return (

            image1.GetSpacing() == image2.GetSpacing()

            and

            image1.GetOrigin() == image2.GetOrigin()

            and

            image1.GetDirection() == image2.GetDirection()

            and

            image1.GetSize() == image2.GetSize()

        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def validate_geometry(
        image: sitk.Image,
        mask: sitk.Image
    ) -> None:
        """
        Raises
        ------
        RuntimeError
            If geometry differs.
        """

        if not ImageUtils.same_geometry(
            image,
            mask
        ):

            raise RuntimeError(

                "Image and mask geometry mismatch.\n"

                f"Image spacing : {image.GetSpacing()}\n"

                f"Mask spacing  : {mask.GetSpacing()}\n"

                f"Image size    : {image.GetSize()}\n"

                f"Mask size     : {mask.GetSize()}"

            )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @staticmethod
    def get_geometry(
        image: sitk.Image
    ) -> dict:
        """
        Return image geometry.
        """

        return {

            "spacing": image.GetSpacing(),

            "origin": image.GetOrigin(),

            "direction": image.GetDirection(),

            "size": image.GetSize()

        }

    # ---------------------------------------------------------
    # Empty Image
    # ---------------------------------------------------------

    @staticmethod
    def is_empty(
        image: sitk.Image
    ) -> bool:

        arr = sitk.GetArrayFromImage(image)

        return arr.size == 0

    # ---------------------------------------------------------
    # Image Statistics
    # ---------------------------------------------------------

    @staticmethod
    def statistics(
        image: sitk.Image
    ) -> dict:

        arr = sitk.GetArrayFromImage(image)

        return {

            "shape": arr.shape,

            "dtype": str(arr.dtype),

            "min": float(arr.min()),

            "max": float(arr.max()),

            "mean": float(arr.mean()),

            "std": float(arr.std())

        }