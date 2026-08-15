"""
metadata.py

Extract DICOM metadata required for preprocessing
and research reproducibility.
"""

from pathlib import Path
import pydicom


class MetadataExtractor:

    @staticmethod
    def extract(ct_folder):

        ct_folder = Path(ct_folder)

        dicom_files = sorted(ct_folder.glob("*.dcm"))

        # Some datasets don't use .dcm extension
        if len(dicom_files) == 0:
            dicom_files = sorted(
                [f for f in ct_folder.iterdir() if f.is_file()]
            )

        if len(dicom_files) == 0:
            raise RuntimeError("No DICOM files found.")

        ds = pydicom.dcmread(
            dicom_files[0],
            stop_before_pixels=True
        )

        metadata = {

            "Manufacturer":
                getattr(ds, "Manufacturer", "Unknown"),

            "ManufacturerModelName":
                getattr(ds, "ManufacturerModelName", "Unknown"),

            "SliceThickness":
                float(getattr(ds, "SliceThickness", 0.0)),

            "PixelSpacing":
                list(getattr(ds, "PixelSpacing", [])),

            "ConvolutionKernel":
                str(getattr(ds, "ConvolutionKernel", "Unknown")),

            "KVP":
                getattr(ds, "KVP", None),

            "Rows":
                ds.Rows,

            "Columns":
                ds.Columns

        }

        return metadata