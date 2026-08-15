"""
dataset_builder.py

Skeleton research-grade DatasetBuilder for LungCancerAI.
NOTE:
This scaffold integrates with the user's preprocessing modules and is
intended as the foundation for the full pipeline.
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import Config
from importlib import util
from pathlib import Path

logger_path = Path(__file__).resolve().parents[1] / "utils" / "logger.py"
spec = util.spec_from_file_location("lungcancer_logger", logger_path)
logger_module = util.module_from_spec(spec)
spec.loader.exec_module(logger_module)
get_logger = logger_module.get_logger

from preprocessing.dicom_reader import NSCLCDicomReader
from preprocessing.seg_loader import SegLoader
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.windowing import CTWindowing
from preprocessing.normalization import Normalizer, NormalizationType
from preprocessing.resample import Resampler
from preprocessing.quality_control import QualityControl
from preprocessing.image_utils import ImageUtils

logger = get_logger(__name__)

@dataclass
class BuildStatistics:
    discovered:int=0
    processed:int=0
    skipped:int=0
    failed:int=0
    warnings:int=0
    failed_patients:List[str]=field(default_factory=list)

class DatasetBuilder:
    def __init__(self):
        self.reader = NSCLCDicomReader(Config.DICOM_DIR)
        self.roi = ROIExtractor()
        self.window = CTWindowing()
        self.normalizer = Normalizer(NormalizationType.Z_SCORE)
        self.resampler = Resampler(Config.TARGET_SPACING)
        self.qc = QualityControl()
        self.stats = BuildStatistics()
        self.output_root = Path(Config.PROCESSED_DIR)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def discover_patients(self):
        pts=self.reader.get_patients()
        self.stats.discovered=len(pts)
        return pts

    def is_processed(self,pid:str)->bool:
        d=self.output_root/pid
        return (d/"image.npy").exists() and (d/"mask.npy").exists() and (d/"metadata.json").exists()

    def process_patient(self, patient_id: str):
        ct_img, _, meta = self.reader.load_patient(patient_id)
        seg_loader = SegLoader.from_patient(Config.DICOM_DIR, patient_id)
        mask_np = seg_loader.load_binary_mask()
        if mask_np.size == 0 or np.count_nonzero(mask_np) == 0:
            raise ValueError("Empty tumor mask")

        mask_img = ImageUtils.numpy_to_sitk(mask_np.astype(np.uint8), ct_img)
        ImageUtils.validate_geometry(ct_img, mask_img)

        ct_img, _ = self.resampler.resample_ct(ct_img)
        mask_img, _ = self.resampler.resample_mask(mask_img)

        ct_np = ImageUtils.sitk_to_numpy(ct_img)
        mask_np = ImageUtils.sitk_to_numpy(mask_img).astype(np.uint8)

        if mask_np.size == 0 or np.count_nonzero(mask_np) == 0:
            raise ValueError("Empty tumor mask")

        ct_np, mask_np, bbox = self.roi.crop(ct_np, mask_np)
        ct_np = self.window.apply(ct_np)
        ct_np = self.normalizer.apply(ct_np)

        report = self.qc.validate(ct_np, mask_np)
        if not report.passed:
            raise RuntimeError("; ".join(report.errors))

        out = self.output_root / patient_id
        out.mkdir(exist_ok=True)

        np.save(out / "image.npy", ct_np)
        np.save(out / "mask.npy", mask_np)

        metadata = {
            "patient_id": patient_id,
            "original_shape": [int(v) for v in meta["shape"]],
            "processed_shape": [int(v) for v in ct_np.shape],
            "bbox": {
                key: int(value) if isinstance(value, (np.integer,)) else value
                for key, value in bbox.__dict__.items()
            },
            "qc_passed": bool(report.passed),
            "warnings": [str(w) for w in report.warnings],
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def build(self):
        for pid in tqdm(self.discover_patients()):
            if self.is_processed(pid):
                self.stats.skipped += 1
                continue
            try:
                self.process_patient(pid)
                self.stats.processed += 1
            except ValueError as exc:
                if "Empty tumor mask" in str(exc):
                    logger.warning("Skipping %s: %s", pid, exc)
                    self.stats.skipped += 1
                else:
                    logger.exception("Failed %s", pid)
                    self.stats.failed += 1
                    self.stats.failed_patients.append(pid)
            except Exception:
                logger.exception("Failed %s", pid)
                self.stats.failed += 1
                self.stats.failed_patients.append(pid)

        logger.info("%s", self.stats)

if __name__=="__main__":
    DatasetBuilder().build()
