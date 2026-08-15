"""
test_gradcam.py

Research Grade GradCAM Test

LungCancerAI
"""

import os
import torch

from datasets.multimodal_dataset import MultiModalDataset
from transforms.ct_transforms import Resize3D

from models.multimodal_model import build_multimodal_model
from explainability.gradcam import GradCAM

# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Device :", device)
print("=" * 70)

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

dataset = MultiModalDataset(
    transform=Resize3D((128, 128, 128))
)

sample = dataset[0]

image = sample["image"].unsqueeze(0).to(device)

clinical = sample["clinical"].unsqueeze(0).to(device)

patient = sample["patient_id"]

print("Patient :", patient)

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = build_multimodal_model(
    clinical_input_dim=5
).to(device)

checkpoint = torch.load(
    "outputs/best/best_model.pth",
    map_location=device,
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif "state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Model Loaded.")

# ----------------------------------------------------------
# GradCAM
# ----------------------------------------------------------

target_layer = model.image_encoder.backbone.layer4

gradcam = GradCAM(

    model=model,

    target_layer=target_layer,

)

# ----------------------------------------------------------
# Histology CAM
# ----------------------------------------------------------

print()

print("Generating Histology CAM...")

histology_cam = gradcam.generate(

    image=image,

    clinical=clinical,

    target_class=None,

    task="histology",

)

overlay = gradcam.overlay(

    image.squeeze().cpu().numpy(),

    histology_cam,

)

os.makedirs(
    "outputs/gradcam",
    exist_ok=True,
)

gradcam.save(

    overlay,

    "outputs/gradcam/histology_cam.png",

)

print("Saved -> outputs/gradcam/histology_cam.png")

# ----------------------------------------------------------
# Stage CAM
# ----------------------------------------------------------

print()

print("Generating Stage CAM...")

stage_cam = gradcam.generate(

    image=image,

    clinical=clinical,

    target_class=None,

    task="stage",

)

overlay = gradcam.overlay(

    image.squeeze().cpu().numpy(),

    stage_cam,

)

gradcam.save(

    overlay,

    "outputs/gradcam/stage_cam.png",

)

print("Saved -> outputs/gradcam/stage_cam.png")

# ----------------------------------------------------------
# Survival CAM
# ----------------------------------------------------------

print()

print("Generating Survival CAM...")

survival_cam = gradcam.generate(

    image=image,

    clinical=clinical,

    target_class=None,

    task="survival",

)

overlay = gradcam.overlay(

    image.squeeze().cpu().numpy(),

    survival_cam,

)

gradcam.save(

    overlay,

    "outputs/gradcam/survival_cam.png",

)

print("Saved -> outputs/gradcam/survival_cam.png")

# ----------------------------------------------------------

print()

print("=" * 70)
print("GradCAM Test Successful")
print("=" * 70)