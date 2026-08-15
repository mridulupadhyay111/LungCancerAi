from pathlib import Path

import numpy as np
import pydicom
import highdicom as hd


class SegLoader:

    def __init__(self, seg_file):
        self.seg_file = Path(seg_file)

    @classmethod
    def from_patient(cls, dataset_root, patient_id):

        dataset_root = Path(dataset_root)

        patient_dir = dataset_root / patient_id

        if not patient_dir.exists():
            raise FileNotFoundError(
                f"Patient not found: {patient_dir}"
            )

        study_dir = next(patient_dir.iterdir())

        seg_dir = None
        # 1. Check folder name starting with SEG
        for p in study_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("SEG"):
                seg_dir = p
                break

        # 2. Check DICOM header Modality == "SEG"
        if seg_dir is None:
            for p in study_dir.iterdir():
                if not p.is_dir():
                    continue
                dcms = list(p.glob("*.dcm"))
                if len(dcms) == 1:
                    try:
                        ds = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
                        if getattr(ds, "Modality", "") == "SEG":
                            seg_dir = p
                            break
                    except Exception:
                        pass

        # 3. Fallback to RTSTRUCT if no SEG found
        if seg_dir is None:
            for p in study_dir.iterdir():
                if not p.is_dir():
                    continue
                if p.name.startswith("RTSTRUCT"):
                    seg_dir = p
                    break
                dcms = list(p.glob("*.dcm"))
                if len(dcms) == 1:
                    try:
                        ds = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
                        if getattr(ds, "Modality", "") == "RTSTRUCT":
                            seg_dir = p
                            break
                    except Exception:
                        pass

        if seg_dir is None:
            raise FileNotFoundError(
                f"No SEG/RTSTRUCT folder found for patient {patient_id}"
            )

        seg_file = next(seg_dir.glob("*.dcm"))

        return cls(seg_file)

    # ---------------------------------------------------------
    # Read SEG
    # ---------------------------------------------------------

    def load(self):
        return hd.seg.segread(self.seg_file)

    # ---------------------------------------------------------
    # Read raw DICOM
    # ---------------------------------------------------------

    def dicom(self):
        return pydicom.dcmread(self.seg_file)

    # ---------------------------------------------------------
    # Print information
    # ---------------------------------------------------------

    def summary(self):

        seg = self.load()
        ds = self.dicom()

        print("=" * 60)
        print("Segmentation Summary")
        print("=" * 60)

        print("Number of Segments :", seg.number_of_segments)
        print("Segment Numbers    :", seg.segment_numbers)
        print("Pixel Array Shape  :", seg.pixel_array.shape)

        print("\nSegment Labels")

        for item in ds.SegmentSequence:

            print("--------------------------------")

            print("Number :", item.SegmentNumber)
            print("Label  :", item.SegmentLabel)

            if hasattr(item, "SegmentDescription"):
                print(
                    "Description:",
                    item.SegmentDescription
                )

        print("=" * 60)

    # ---------------------------------------------------------
    # Return labels
    # ---------------------------------------------------------

    def get_segment_labels(self):

        ds = self.dicom()

        return [
            item.SegmentLabel
            for item in ds.SegmentSequence
        ]

    # ---------------------------------------------------------
    # Load Primary Tumor Mask
    # ---------------------------------------------------------

    def load_binary_mask(self):
        """
        Automatically identify the primary tumor
        (GTV / Neoplasm / Tumor) and return it as
        a binary mask.
        """

        seg = self.load()
        ds = self.dicom()

        pixels = seg.pixel_array

        n_segments = seg.number_of_segments

        z = pixels.shape[0] // n_segments

        masks = [
            pixels[i * z:(i + 1) * z]
            for i in range(n_segments)
        ]

        keywords = [
            "gtv",
            "tumor",
            "tumour",
            "neoplasm",
            "primary",
            "gross"
        ]

        tumor_index = None

        print("\nDetected Segments")
        print("-" * 60)

        for i, item in enumerate(ds.SegmentSequence):

            label = getattr(
                item,
                "SegmentLabel",
                ""
            )

            description = getattr(
                item,
                "SegmentDescription",
                ""
            )

            print(
                f"{i+1}. {label} | {description}"
            )

            text = (
                label + " " + description
            ).lower()

            if any(k in text for k in keywords):
                tumor_index = i

        if tumor_index is None:
            non_empty = [
                i for i, mask in enumerate(masks)
                if np.count_nonzero(mask) > 0
            ]
            if non_empty:
                tumor_index = non_empty[0]
            else:
                raise RuntimeError(
                    "Primary tumor segment not found."
                )

        print(
            f"\nTumor Segment Selected : {tumor_index + 1}"
        )

        tumor = masks[tumor_index].astype(np.uint8)

        return tumor