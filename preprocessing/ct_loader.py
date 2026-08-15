from pathlib import Path
import pydicom
import numpy as np


class CTLoader:

    def __init__(self, ct_folder):
        self.ct_folder = Path(ct_folder)

    def load(self):
        # Read all DICOM slices
        files = list(self.ct_folder.glob("*.dcm"))

        slices = [pydicom.dcmread(f) for f in files]

        # Sort slices by z position
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        # Stack slices
        volume = np.stack([s.pixel_array for s in slices])

        return volume, slices

    def summary(self):

        volume, slices = self.load()

        print("=" * 60)
        print("CT Summary")
        print("=" * 60)

        print("Shape :", volume.shape)

        print("Voxel spacing :",
              slices[0].PixelSpacing,
              slices[0].SliceThickness)

        print("Patient :", slices[0].PatientID)

        print("HU range :",
              volume.min(),
              volume.max())

        print("=" * 60)