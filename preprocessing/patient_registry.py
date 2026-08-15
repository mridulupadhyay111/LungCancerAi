"""
patient_registry.py

Research-grade Patient Registry Builder
for the NSCLC-Radiomics Dataset.

Author: LungCancerAI
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import logging

import pandas as pd
import pydicom


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Data Class
# ------------------------------------------------------------

@dataclass
class PatientRecord:

    patient_id: str

    study_uid: str

    ct_path: Optional[str]

    rtstruct_path: Optional[str]

    seg_paths: List[str]

    sr_paths: List[str]

    slice_count: int

    spacing_x: float

    spacing_y: float

    spacing_z: float

    manufacturer: str

    model: str

    clinical_available: bool

    status: str


# ------------------------------------------------------------
# Registry Builder
# ------------------------------------------------------------

class PatientRegistry:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

        self.records = []

    # --------------------------------------------------------

    def discover_patients(self):

        patients = sorted(

            p for p in self.dataset_root.iterdir()

            if p.is_dir() and p.name.startswith("LUNG1")

        )

        logger.info(f"Found {len(patients)} patients")

        return patients

    # --------------------------------------------------------

    @staticmethod
    def find_folder(study_path: Path, prefix: str):

        folders = [

            f for f in study_path.iterdir()

            if f.is_dir() and f.name.startswith(prefix)

        ]

        if len(folders) == 0:
            return None

        return folders[0]

    # --------------------------------------------------------

    @staticmethod
    def find_all(study_path: Path, prefix: str):

        return [

            str(f)

            for f in study_path.iterdir()

            if f.is_dir() and f.name.startswith(prefix)

        ]

    # --------------------------------------------------------

    @staticmethod
    def count_slices(ct_folder: Path):

        dicoms = list(ct_folder.glob("*"))

        return len([f for f in dicoms if f.is_file()])

    # --------------------------------------------------------

    @staticmethod
    def read_metadata(ct_folder: Path):

        files = [

            f for f in ct_folder.iterdir()

            if f.is_file()

        ]

        ds = pydicom.dcmread(

            files[0],

            stop_before_pixels=True

        )

        spacing = ds.PixelSpacing

        spacing_z = getattr(

            ds,

            "SliceThickness",

            0.0

        )

        manufacturer = getattr(

            ds,

            "Manufacturer",

            "Unknown"

        )

        model = getattr(

            ds,

            "ManufacturerModelName",

            "Unknown"

        )

        return (

            float(spacing[0]),

            float(spacing[1]),

            float(spacing_z),

            manufacturer,

            model

        )

    # --------------------------------------------------------

    def process_patient(self, patient_folder):

        study_folders = [

            f for f in patient_folder.iterdir()

            if f.is_dir()

        ]

        if len(study_folders) == 0:

            logger.warning(

                f"{patient_folder.name}: no study"

            )

            return

        study = study_folders[0]

        ct = self.find_folder(

            study,

            "CT_"

        )

        rt = self.find_folder(

            study,

            "RTSTRUCT"

        )

        seg = self.find_all(

            study,

            "SEG"

        )

        sr = self.find_all(

            study,

            "SR"

        )

        if ct is None:

            status = "Missing CT"

            record = PatientRecord(

                patient_id=patient_folder.name,

                study_uid=study.name,

                ct_path=None,

                rtstruct_path=None,

                seg_paths=[],

                sr_paths=[],

                slice_count=0,

                spacing_x=0,

                spacing_y=0,

                spacing_z=0,

                manufacturer="",

                model="",

                clinical_available=False,

                status=status

            )

            self.records.append(record)

            return

        sx, sy, sz, manufacturer, model = self.read_metadata(ct)

        slices = self.count_slices(ct)

        record = PatientRecord(

            patient_id=patient_folder.name,

            study_uid=study.name,

            ct_path=str(ct),

            rtstruct_path=str(rt) if rt else "",

            seg_paths=seg,

            sr_paths=sr,

            slice_count=slices,

            spacing_x=sx,

            spacing_y=sy,

            spacing_z=sz,

            manufacturer=manufacturer,

            model=model,

            clinical_available=False,

            status="Ready"

        )

        self.records.append(record)

    # --------------------------------------------------------

    def build(self):

        patients = self.discover_patients()

        for patient in patients:

            logger.info(

                f"Processing {patient.name}"

            )

            self.process_patient(patient)

        df = pd.DataFrame(

            [

                asdict(r)

                for r in self.records

            ]

        )

        return df

    # --------------------------------------------------------

    @staticmethod
    def save(df, output_dir):

        output_dir = Path(output_dir)

        output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        csv_path = output_dir / "patient_registry.csv"

        parquet_path = output_dir / "patient_registry.parquet"

        df.to_csv(

            csv_path,

            index=False

        )

        df.to_parquet(

            parquet_path,

            index=False

        )

        logger.info(f"CSV saved : {csv_path}")

        logger.info(f"Parquet saved : {parquet_path}")