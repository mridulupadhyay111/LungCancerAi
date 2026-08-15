"""
quality_control.py

Research-grade Quality Control module.

Stores preprocessing metadata for every patient.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


class QualityControl:

    def __init__(self, output_directory):

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_report(

        self,

        patient_id,

        metadata,

        hu_statistics,

        processing_stage,

        extra=None

    ):

        report = {

            "patient_id": patient_id,

            "processing_stage": processing_stage,

            "timestamp": datetime.now().isoformat(),

            "metadata": {

                "shape": list(metadata["shape"]),

                "spacing": list(metadata["spacing"]),

                "origin": list(metadata["origin"]),

                "direction": list(metadata["direction"])

            },

            "hu_statistics": {

                "minimum": hu_statistics.minimum,

                "maximum": hu_statistics.maximum,

                "mean": hu_statistics.mean,

                "std": hu_statistics.std,

                "air_percentage": float(
                    hu_statistics.air_percentage
                ),

                "soft_tissue_percentage": float(
                    hu_statistics.soft_tissue_percentage
                ),

                "bone_percentage": float(
                    hu_statistics.bone_percentage
                ),

                "valid_hu": hu_statistics.valid_hu

            }

        }

        if extra is not None:

            report["extra"] = extra

        output_file = (

            self.output_directory

            / f"{patient_id}.json"

        )

        with open(

            output_file,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                report,

                f,

                indent=4

            )