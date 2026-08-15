"""
quality_control.py

Research-grade preprocessing quality control.

Author: LungCancerAI
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QCReport:

    passed: bool

    errors: list

    warnings: list


class QualityControl:

    def validate(
        self,
        ct: np.ndarray,
        mask: np.ndarray
    ):

        errors = []
        warnings = []

        # ------------------------
        # Shape check
        # ------------------------
        if ct.shape != mask.shape:

            errors.append(
                "CT and Mask shape mismatch."
            )

        # ------------------------
        # Empty mask
        # ------------------------
        if mask.sum() == 0:

            errors.append(
                "Mask is empty."
            )

        # ------------------------
        # Binary mask
        # ------------------------
        values = np.unique(mask)

        if not np.all(np.isin(values, [0, 1])):

            errors.append(
                f"Mask contains values {values}"
            )

        # ------------------------
        # NaN values
        # ------------------------
        if np.isnan(ct).any():

            errors.append(
                "CT contains NaN."
            )

        # ------------------------
        # Infinite values
        # ------------------------
        if np.isinf(ct).any():

            errors.append(
                "CT contains Inf."
            )

        # ------------------------
        # Extremely small tumors
        # ------------------------
        tumor_voxels = int(mask.sum())

        if tumor_voxels < 100:

            warnings.append(
                f"Tumor very small ({tumor_voxels} voxels)"
            )

        return QCReport(

            passed=len(errors) == 0,

            errors=errors,

            warnings=warnings

        )