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
# ============================================================
# Predict Function
# ============================================================

@torch.no_grad()
def predict_patient(index):

    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(DEVICE)

    clinical = sample["clinical"].unsqueeze(0).to(DEVICE)

    outputs = model.predict(

        image,

        clinical,

    )

    histology = torch.argmax(

        outputs["histology_logits"],

        dim=1,

    ).item()

    stage = torch.argmax(

        outputs["stage_logits"],

        dim=1,

    ).item()

    survival = outputs[
        "survival_prediction"
    ].item()

    print()

    print("=" * 70)

    print("Prediction")

    print("=" * 70)

    print("Patient :", sample["patient_id"])

    print("Histology :", histology)

    print("Stage :", stage)

    print("Survival Score :", round(survival, 4))

    print("=" * 70)

    return outputs


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    predict_patient(15)
    