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
# Dataset
# ==========================================================

dataset = MultiModalDataset()

print()

print("Dataset Size :", len(dataset))
# ==========================================================
# Train Validation Split
# ==========================================================

train_size = int(

    0.8 * len(dataset)

)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(

    dataset,

    [train_size, val_size],

    generator=torch.Generator().manual_seed(42),

)

print()

print("Train :", len(train_dataset))

print("Validation :", len(val_dataset))
# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=2,

    shuffle=True,

    num_workers=0,

    pin_memory=True,

)

val_loader = DataLoader(

    val_dataset,

    batch_size=2,

    shuffle=False,

    num_workers=0,

    pin_memory=True,

)
# ==========================================================
# Build Model
# ==========================================================

model = build_multimodal_model(

    clinical_input_dim=dataset.num_features,

)

model = model.to(device)

print()

print("=" * 70)

print("Model Built Successfully")

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
# Loss
# ==========================================================

criterion = MultiTaskLoss()

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

    gradient_clip=5.0,

    accumulation_steps=1,

    early_stopping_patience=15,

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