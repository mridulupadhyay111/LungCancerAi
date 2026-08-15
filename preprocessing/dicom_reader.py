"""
dicom_reader.py

Research-grade DICOM reader for the NSCLC-Radiomics dataset.

Features
--------
- Automatically finds CT series
- Reads DICOM using SimpleITK
- Returns CT image, numpy array and metadata
- Accepts either patient ID (str) or Path object

Author: LungCancerAI
"""

from pathlib import Path

import SimpleITK as sitk


class NSCLCDicomReader:

    def __init__(self, dataset_root):
        """
        Parameters
        ----------
        dataset_root : str or Path

        Example
        -------
        D:/LungCancerAI/data/raw/dicom/nsclc_radiomics
        """

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset folder not found:\n{self.dataset_root}"
            )

    # --------------------------------------------------------
    # List all patients
    # --------------------------------------------------------
    def get_patients(self):

        patients = sorted(
            [
                p.name
                for p in self.dataset_root.iterdir()
                if p.is_dir() and p.name.startswith("LUNG1")
            ]
        )

        return patients

    # --------------------------------------------------------
    # Locate CT folder
    # --------------------------------------------------------
    def find_ct_folder(self, patient_path: Path):

        uid_folders = [
            p for p in patient_path.iterdir()
            if p.is_dir()
        ]

        if len(uid_folders) == 0:
            raise RuntimeError(
                f"No UID folder found for {patient_path.name}"
            )

        uid_folder = uid_folders[0]

        ct_folders = [
            p for p in uid_folder.iterdir()
            if p.is_dir() and p.name.startswith("CT_")
        ]

        if len(ct_folders) == 0:
            # Fallback for raw TCIA folder names without CT_ prefix
            for p in uid_folder.iterdir():
                if not p.is_dir():
                    continue
                dcm_files = list(p.glob("*.dcm"))
                if len(dcm_files) > 5:  # CT 3D volumes have multiple slice files
                    ct_folders.append(p)
                    break
                elif len(dcm_files) == 1:
                    try:
                        import pydicom
                        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                        if getattr(ds, "Modality", "") == "CT":
                            ct_folders.append(p)
                            break
                    except Exception:
                        pass

        if len(ct_folders) == 0:
            raise RuntimeError(
                f"No CT folder found for {patient_path.name}"
            )

        return ct_folders[0]

    # --------------------------------------------------------
    # Load patient CT
    # --------------------------------------------------------
    def load_patient(self, patient):

        """
        Parameters
        ----------
        patient : str or Path

        Examples
        --------
        load_patient("LUNG1-100")

        OR

        load_patient(Path(...))
        """

        # -------------------------------
        # Convert to Path
        # -------------------------------
        if isinstance(patient, str):

            patient_path = self.dataset_root / patient

        else:

            patient_path = Path(patient)

        if not patient_path.exists():

            raise FileNotFoundError(
                f"Patient folder not found:\n{patient_path}"
            )

        # -------------------------------
        # Find CT folder
        # -------------------------------
        ct_folder = self.find_ct_folder(patient_path)

        reader = sitk.ImageSeriesReader()

        series_ids = reader.GetGDCMSeriesIDs(str(ct_folder))

        if len(series_ids) == 0:

            raise RuntimeError(
                f"No DICOM series found in\n{ct_folder}"
            )

        dicom_files = reader.GetGDCMSeriesFileNames(
            str(ct_folder),
            series_ids[0]
        )

        reader.SetFileNames(dicom_files)

        image = reader.Execute()

        volume = sitk.GetArrayFromImage(image)

        metadata = {

            "spacing": image.GetSpacing(),

            "origin": image.GetOrigin(),

            "direction": image.GetDirection(),

            "shape": volume.shape,

            "num_slices": volume.shape[0],

            "height": volume.shape[1],

            "width": volume.shape[2]

        }

        return image, volume, metadata

    # --------------------------------------------------------
    # Print dataset summary
    # --------------------------------------------------------
    def summary(self):

        patients = self.get_patients()

        print("=" * 60)
        print("NSCLC-Radiomics Dataset")
        print("=" * 60)

        print("Dataset :", self.dataset_root)

        print("Patients:", len(patients))

        print("First 5 Patients")

        for p in patients[:5]:
            print("   ", p)

        logger.info(f"Dataset : {self.dataset_root}")
        logger.info(f"Patients : {len(patients)}")