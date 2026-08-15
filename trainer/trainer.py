"""
trainer.py

Research Grade Trainer

Supports

• AMP
• Gradient Accumulation
• TensorBoard
• EarlyStopping
• Checkpointing
• Resume
• Validation
• Scheduler
• Gradient Clipping
• Metrics
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.cuda.amp import (
    autocast,
    GradScaler,
)

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


@dataclass
class TrainerConfig:

    epochs: int = 100

    save_dir: str = "outputs"

    experiment_name: str = "multimodal"

    mixed_precision: bool = False

    gradient_clip: float = 5.0

    accumulation_steps: int = 1

    early_stopping_patience: int = 15

    save_best_only: bool = True

    log_every: int = 10

    device: str = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )
class Trainer:

    def __init__(

        self,

        model,

        optimizer,

        scheduler,

        criterion,

        config: TrainerConfig,

    ):

        self.model = model.to(

            config.device

        )

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        self.config = config

        self.device = config.device

        self.scaler = GradScaler(

            enabled=config.mixed_precision

        )

        self.best_loss = 1e9

        self.best_epoch = -1

        self.epochs_without_improvement = 0

        self.global_step = 0

        self.writer = SummaryWriter(

            Path(config.save_dir)

            / "tensorboard"

        )

        Path(

            config.save_dir

        ).mkdir(

            parents=True,

            exist_ok=True,

        )

        Path(

            config.save_dir,

            "best",

        ).mkdir(

            parents=True,

            exist_ok=True,

        )

        Path(

            config.save_dir,

            "checkpoints",

        ).mkdir(

            parents=True,

            exist_ok=True,

        )

        print("=" * 70)

        print("Trainer Initialized")

        print("=" * 70)

        print("Device :", self.device)

        print("AMP :", config.mixed_precision)

        print("Save :", config.save_dir)

        print("=" * 70)
        # ============================================================
    # Train One Epoch
    # ============================================================

    def train_one_epoch(

        self,

        train_loader,

        epoch,

    ):

        self.model.train()

        running_loss = 0.0

        running_histology = 0.0

        running_stage = 0.0

        running_survival = 0.0

        progress = tqdm(

            train_loader,

            desc=f"Train {epoch+1}",

            leave=False,

        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):

            image = batch["image"].to(self.device)

            clinical = batch["clinical"].to(self.device)

            histology = batch["histology"].to(self.device)

            stage = batch["stage"].to(self.device)

            survival = batch["survival"].to(self.device)

            event = batch["event"].to(self.device)

            with autocast(enabled=self.config.mixed_precision):

                outputs = self.model(

                    image,

                    clinical,

                )
                targets = {
                    "histology": histology,
                    "stage": stage,
                    "survival": survival,
                }

                losses = self.criterion(outputs, targets)
                loss = losses["loss"] / self.config.accumulation_steps
                self.scaler.scale(loss).backward()

            if (

                step + 1

            ) % self.config.accumulation_steps == 0:

                self.scaler.unscale_(

                    self.optimizer

                )

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    self.config.gradient_clip,

                )

                self.scaler.step(

                    self.optimizer

                )

                self.scaler.update()

                self.optimizer.zero_grad()

                if self.scheduler is not None:

                    if hasattr(

                        self.scheduler,

                        "step",

                    ):

                        self.scheduler.step()

            running_loss += losses["loss"].item()

            running_histology += losses["histology_loss"].item()

            running_stage += losses["stage_loss"].item()

            running_survival += losses["survival_loss"].item()

            progress.set_postfix(

                {

                    "loss":

                        f"{running_loss/(step+1):.4f}"

                }

            )

            self.global_step += 1

        epoch_loss = (

            running_loss

            /

            len(train_loader)

        )

        metrics = {

            "loss":

                epoch_loss,

            "histology_loss":

                running_histology

                /

                len(train_loader),

            "stage_loss":

                running_stage

                /

                len(train_loader),

            "survival_loss":

                running_survival

                /

                len(train_loader),

        }

        return metrics
        # ============================================================
    # Validation
    # ============================================================

    @torch.no_grad()

    def validate(

        self,

        val_loader,

    ):

        self.model.eval()

        total_loss = 0.0

        histology_loss = 0.0

        stage_loss = 0.0

        survival_loss = 0.0

        progress = tqdm(

            val_loader,

            desc="Validation",

            leave=False,

        )

        for batch in progress:

            image = batch["image"].to(self.device)

            clinical = batch["clinical"].to(self.device)

            histology = batch["histology"].to(self.device)

            stage = batch["stage"].to(self.device)

            survival = batch["survival"].to(self.device)

            event = batch["event"].to(self.device)

            outputs = self.model(

                image,

                clinical,

            )

            targets = {
                "histology": histology,
                "stage": stage,
                "survival": survival,
            }
            losses = self.criterion(outputs, targets)

            total_loss += losses["loss"].item()

            histology_loss += losses["histology_loss"].item()

            stage_loss += losses["stage_loss"].item()

            survival_loss += losses["survival_loss"].item()

        return {

            "loss":

                total_loss / len(val_loader),

            "histology_loss":

                histology_loss / len(val_loader),

            "stage_loss":

                stage_loss / len(val_loader),

            "survival_loss":

                survival_loss / len(val_loader),

        }
        # ============================================================
    # Save Checkpoint
    # ============================================================

    def save_checkpoint(

        self,

        epoch,

        val_loss,

        best=False,

    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "best_loss": self.best_loss,

            "global_step": self.global_step,

        }

        if self.scheduler is not None:

            checkpoint["scheduler_state_dict"] = (

                self.scheduler.state_dict()

            )

        save_path = (

            Path(self.config.save_dir)

            / "checkpoints"

            / f"epoch_{epoch+1}.pth"

        )

        torch.save(

            checkpoint,

            save_path,

        )

        if best:

            best_path = (

                Path(self.config.save_dir)

                / "best"

                / "best_model.pth"

            )

            torch.save(

                checkpoint,

                best_path,

            )

            print()

            print("=" * 70)

            print("Best Model Updated")

            print(best_path)

            print("=" * 70)

    # ============================================================
    # Load Checkpoint
    # ============================================================

    def load_checkpoint(

        self,

        checkpoint_path,

        load_optimizer=True,

    ):

        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device,

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        if load_optimizer:

            self.optimizer.load_state_dict(

                checkpoint["optimizer_state_dict"]

            )

            if (

                self.scheduler is not None

                and

                "scheduler_state_dict"

                in checkpoint

            ):

                self.scheduler.load_state_dict(

                    checkpoint["scheduler_state_dict"]

                )

        self.best_loss = checkpoint.get(

            "best_loss",

            1e9,

        )

        self.global_step = checkpoint.get(

            "global_step",

            0,

        )

        print()

        print("=" * 70)

        print("Checkpoint Loaded")

        print(checkpoint_path)

        print("=" * 70)

        return checkpoint.get(

            "epoch",

            0,

        )
        # ============================================================
    # TensorBoard Logging
    # ============================================================

    def log_metrics(

        self,

        train_metrics,

        val_metrics,

        epoch,

    ):

        for key, value in train_metrics.items():

            self.writer.add_scalar(

                f"train/{key}",

                value,

                epoch,

            )

        for key, value in val_metrics.items():

            self.writer.add_scalar(

                f"val/{key}",

                value,

                epoch,

            )

        lr = self.optimizer.param_groups[0]["lr"]

        self.writer.add_scalar(

            "learning_rate",

            lr,

            epoch,

        )

    # ============================================================
    # Early Stopping
    # ============================================================

    def early_stopping(

        self,

        val_loss,

    ):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.epochs_without_improvement = 0

            return False, True

        self.epochs_without_improvement += 1

        stop = (

            self.epochs_without_improvement

            >=

            self.config.early_stopping_patience

        )

        return stop, False
        # ============================================================
    # Main Training Loop
    # ============================================================

    def fit(

        self,

        train_loader,

        val_loader,

        epochs,

    ):

        print()

        print("=" * 70)

        print("Starting Training")

        print("=" * 70)

        start_time = time.time()

        for epoch in range(epochs):

            print()

            print("-" * 70)

            print(f"Epoch {epoch+1}/{epochs}")

            print("-" * 70)

            train_metrics = self.train_one_epoch(

                train_loader,

                epoch,

            )

            val_metrics = self.validate(

                val_loader,

            )

            self.log_metrics(

                train_metrics,

                val_metrics,

                epoch,

            )

            print()

            print(

                f"Train Loss : {train_metrics['loss']:.4f}"

            )

            print(

                f"Val Loss   : {val_metrics['loss']:.4f}"

            )

            stop, best = self.early_stopping(

                val_metrics["loss"]

            )

            self.save_checkpoint(

                epoch,

                val_metrics["loss"],

                best,

            )

            if stop:

                print()

                print("=" * 70)

                print("Early Stopping Triggered")

                print("=" * 70)

                break

        elapsed = (

            time.time()

            -

            start_time

        ) / 60

        print()

        print("=" * 70)

        print("Training Finished")

        print(f"Best Validation Loss : {self.best_loss:.4f}")

        print(f"Training Time : {elapsed:.2f} minutes")

        print("=" * 70)
        # ============================================================
    # Evaluate
    # ============================================================

    @torch.no_grad()

    def evaluate(

        self,

        data_loader,

    ):

        metrics = self.validate(

            data_loader,

        )

        print()

        print("=" * 70)

        print("Evaluation")

        print("=" * 70)

        for k, v in metrics.items():

            print(

                f"{k:20s}: {v:.4f}"

            )

        print("=" * 70)

        return metrics

    # ============================================================
    # Close
    # ============================================================

    def close(self):

        self.writer.close()


# ============================================================
# Self Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("Trainer Module Imported Successfully")

    print("=" * 70)                                            