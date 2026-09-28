"""
predict.py

Research-grade inference script for LungCancerAI.

Predicts:
1. Histology
2. Stage
3. Survival Risk
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs.config import Config
from datasets.multimodal_dataset import MultiModalDataset
from models.multimodal_model import build_multimodal_model
from transforms.ct_transforms import Resize3D

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Inference Device :", DEVICE)
print("=" * 60)

# ============================================================
# Dataset
# ============================================================

dataset = MultiModalDataset(
    processed_dir=Config.PROCESSED_DIR,
    clinical_dir=Config.PROCESSED_DIR / "clinical",
    transform=Resize3D((128, 128, 128)),
    use_mask=True,
)

# ============================================================
# Model
# ============================================================

model = build_multimodal_model(
    clinical_input_dim=5,
    image_embedding_dim=512,
    clinical_embedding_dim=256,
    fusion_dim=512,
)

checkpoint = Path("outputs") / "best" / "best_model.pth"

checkpoint = torch.load(
    checkpoint,
    map_location=DEVICE,
)

if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("Best model loaded.")
import argparse
from pathlib import Path
import torch

from deployment.predictor import LungCancerPredictor

def main():
    parser = argparse.ArgumentParser(description="LungCancerAI Predict Script")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to predict")
    parser.add_argument("--patient_id", type=str, default=None, help="Patient ID to predict")
    args = parser.parse_args()

    predictor = LungCancerPredictor()
    
    target = args.patient_id if args.patient_id else args.index
    print(f"Running inference for target: {target}")
    
    res = predictor.predict(target)
    
    print()
    print("=" * 70)
    print("Prediction Result")
    print("=" * 70)
    print("Patient ID     :", res["patient_id"])
    print("Histology      :", res["histology_label"])
    print("Histology Probs:", [round(float(p), 4) for p in res["histology_probs"]])
    print("Stage          :", res["stage_label"])
    print("Stage Probs    :", [round(float(p), 4) for p in res["stage_probs"]])
    print("Survival Days  :", res["survival_days"])
    print("Survival Years :", res["survival_years"])
    print("=" * 70)

if __name__ == "__main__":
    main()