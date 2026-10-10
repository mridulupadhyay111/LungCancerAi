"""
train.py

Research-grade training pipeline for LungCancerAI
"""
print("USING TRAIN.PY:", __file__)
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from configs.config import Config

from datasets.multimodal_dataset import MultiModalDataset

from models.multimodal_model import build_multimodal_model

from losses.multitask_loss import MultiTaskLoss

from optimizers.optimizer import (
    build_optimizer,
    OptimizerConfig,
)

from schedulers.scheduler import (
    SchedulerBuilder,
    SchedulerConfig,
    SchedulerType,
)

from trainer.trainer import (
    Trainer,
    TrainerConfig,
)
# ==========================================================
# Random Seed
# ==========================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything()
# ==========================================================
# Device
# ==========================================================
# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("Device :", device)
print("=" * 70)
# ==========================================================
# Datasets (Separate Train & Validation for Augmentations)
# ==========================================================

full_train_dataset = MultiModalDataset(training=True)
full_val_dataset = MultiModalDataset(training=False)

dataset_size = len(full_train_dataset)
train_size = int(0.8 * dataset_size)

np.random.seed(42)
indices = np.random.permutation(dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

print()
print("Dataset Size :", dataset_size)
print("Train        :", len(train_dataset))
print("Validation   :", len(val_dataset))

# ==========================================================
# Class Weights & Balanced Sampling
# ==========================================================

train_df = full_train_dataset.data.iloc[train_indices]
train_histology = train_df["histology"].values
train_stage = train_df["stage"].values

num_histology_classes = len(np.unique(full_train_dataset.data["histology"]))
num_stage_classes = len(np.unique(full_train_dataset.data["stage"]))

histology_counts = np.bincount(train_histology, minlength=max(5, num_histology_classes))
stage_counts = np.bincount(train_stage, minlength=max(4, num_stage_classes))

histology_weights = torch.tensor(
    [len(train_histology) / (float(len(histology_counts)) * max(1, c)) for c in histology_counts],
    dtype=torch.float32,
).to(device)

stage_weights = torch.tensor(
    [len(train_stage) / (float(len(stage_counts)) * max(1, c)) for c in stage_counts],
    dtype=torch.float32,
).to(device)

sample_weights = [
    float(histology_weights[h].cpu() * stage_weights[s].cpu())
    for h, s in zip(train_histology, train_stage)
]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4 if device.type == "cuda" else 2,
    sampler=sampler,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4 if device.type == "cuda" else 2,
    shuffle=False,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
)

# ==========================================================
# Build Model
# ==========================================================

model = build_multimodal_model(
    clinical_input_dim=full_train_dataset.num_features,
    num_histology_classes=len(histology_counts),
    num_stage_classes=len(stage_counts),
)

model = model.to(device)

print()
print("=" * 70)
print("Multimodal Model Built Successfully")
print("=" * 70)

# ==========================================================
# Optimizer
# ==========================================================

optimizer_config = OptimizerConfig()

optimizer = build_optimizer(
    model,
    optimizer_config,
)

print()
print("=" * 70)
print("Optimizer Built Successfully")
print("=" * 70)

# ==========================================================
# Scheduler
# ==========================================================

scheduler_config = SchedulerConfig(
    scheduler=SchedulerType.COSINE,
    epochs=100,
    warmup_epochs=5,
    min_lr=1e-6,
)

scheduler_builder = SchedulerBuilder(
    optimizer=optimizer,
    config=scheduler_config,
)

scheduler = scheduler_builder.build()

print()
print("=" * 70)
print("Scheduler Built Successfully")
print("=" * 70)

# ==========================================================
# Loss Function with Balanced Class Weights
# ==========================================================

criterion = MultiTaskLoss(
    histology_class_weights=histology_weights,
    stage_class_weights=stage_weights,
)

print()
print("=" * 70)
print("Loss Built Successfully")
print("=" * 70)
# ==========================================================
# Trainer
# ==========================================================

trainer_config = TrainerConfig(

    epochs=100,

    save_dir="outputs",

    experiment_name="LungCancerAI",

    mixed_precision=False,      # CPU training

    gradient_clip=1.0,

    accumulation_steps=8,

    early_stopping_patience=25,

    save_best_only=True,

    log_every=10,

    device=str(device),

)

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    scheduler=scheduler,

    criterion=criterion,

    config=trainer_config,

)

print()
print("=" * 70)
print("Trainer Built Successfully")
print("=" * 70)
# ==========================================================
# Resume (Optional)
# ==========================================================

resume = False

checkpoint = Path("outputs") / "best" / "best_model.pth"

if resume and checkpoint.exists():

    trainer.load_checkpoint(
        checkpoint,
        load_optimizer=True,
    )

# ==========================================================
# Trainer
# ==========================================================
# Training
# ==========================================================

print()
print("=" * 70)
print("Starting Training")
print("=" * 70)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=trainer_config.epochs,
)

# ==========================================================
# Final Evaluation
# ==========================================================

print()
print("=" * 70)
print("Evaluating Best Model")
print("=" * 70)

best_model = Path("outputs") / "best" / "best_model.pth"

if best_model.exists():

    trainer.load_checkpoint(
        best_model,
        load_optimizer=False,
    )

trainer.evaluate(val_loader)
trainer.close()

print()
print("=" * 70)
print("Training Completed Successfully")
print("=" * 70)