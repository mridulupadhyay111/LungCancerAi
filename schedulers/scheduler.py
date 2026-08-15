"""
scheduler.py

Research-grade learning rate scheduler module
for LungCancerAI.

Supports
--------
1. Cosine Annealing
2. Cosine Warm Restarts
3. OneCycle
4. StepLR
5. MultiStepLR
6. ExponentialLR
7. ReduceLROnPlateau
8. Linear Warmup
9. Warmup + Cosine
10. Resume Compatible

Author
------
LungCancerAI
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    LinearLR,
)


# ============================================================
# Scheduler Types
# ============================================================

class SchedulerType(Enum):

    COSINE = "cosine"

    COSINE_RESTART = "cosine_restart"

    ONE_CYCLE = "one_cycle"

    STEP = "step"

    MULTISTEP = "multistep"

    EXPONENTIAL = "exponential"

    PLATEAU = "plateau"

    WARMUP_COSINE = "warmup_cosine"

    NONE = "none"


# ============================================================
# Scheduler Configuration
# ============================================================

@dataclass
class SchedulerConfig:

    scheduler: SchedulerType = SchedulerType.WARMUP_COSINE

    epochs: int = 100

    steps_per_epoch: int = 100

    max_lr: float = 1e-3

    min_lr: float = 1e-6

    warmup_epochs: int = 5

    step_size: int = 30

    gamma: float = 0.1

    milestones: tuple = (40, 70, 90)

    patience: int = 8

    factor: float = 0.2

    threshold: float = 1e-4

    cooldown: int = 2

    eta_min: float = 1e-6

    restart_period: int = 20

    restart_mult: int = 2

    verbose: bool = True

    # ============================================================
# Scheduler Builder
# ============================================================

class SchedulerBuilder:

    def __init__(

        self,

        optimizer,

        config: SchedulerConfig,

    ):

        self.optimizer = optimizer

        self.config = config

    # ----------------------------------------------------------
    # Build Scheduler
    # ----------------------------------------------------------

    def build(self):

        scheduler_type = self.config.scheduler

        if scheduler_type == SchedulerType.NONE:

            return None

        if scheduler_type == SchedulerType.COSINE:

            return self.build_cosine()

        if scheduler_type == SchedulerType.COSINE_RESTART:

            return self.build_cosine_restart()

        if scheduler_type == SchedulerType.ONE_CYCLE:

            return self.build_onecycle()

        if scheduler_type == SchedulerType.STEP:

            return self.build_step()

        if scheduler_type == SchedulerType.MULTISTEP:

            return self.build_multistep()

        if scheduler_type == SchedulerType.EXPONENTIAL:

            return self.build_exponential()

        if scheduler_type == SchedulerType.PLATEAU:

            return self.build_plateau()

        if scheduler_type == SchedulerType.WARMUP_COSINE:

            return self.build_warmup_cosine()

        raise ValueError(

            f"Unknown scheduler {scheduler_type}"

        )

    # ----------------------------------------------------------
    # Cosine Annealing
    # ----------------------------------------------------------

    def build_cosine(self):

        return CosineAnnealingLR(

            optimizer=self.optimizer,

            T_max=self.config.epochs,

            eta_min=self.config.eta_min,

        )

    # ----------------------------------------------------------
    # Cosine Warm Restarts
    # ----------------------------------------------------------

    def build_cosine_restart(self):

        return CosineAnnealingWarmRestarts(

            optimizer=self.optimizer,

            T_0=self.config.restart_period,

            T_mult=self.config.restart_mult,

            eta_min=self.config.eta_min,

        )

    # ----------------------------------------------------------
    # Step LR
    # ----------------------------------------------------------

    def build_step(self):

        return StepLR(

            optimizer=self.optimizer,

            step_size=self.config.step_size,

            gamma=self.config.gamma,

        )

    # ----------------------------------------------------------
    # MultiStep LR
    # ----------------------------------------------------------

    def build_multistep(self):

        return MultiStepLR(

            optimizer=self.optimizer,

            milestones=list(self.config.milestones),

            gamma=self.config.gamma,

        )

    # ----------------------------------------------------------
    # Exponential LR
    # ----------------------------------------------------------

    def build_exponential(self):

        return ExponentialLR(

            optimizer=self.optimizer,

            gamma=self.config.gamma,

        )
        # ----------------------------------------------------------
    # Reduce LR on Plateau
    # ----------------------------------------------------------

    def build_plateau(self):

        return ReduceLROnPlateau(

            optimizer=self.optimizer,

            mode="min",

            factor=self.config.factor,

            patience=self.config.patience,

            threshold=self.config.threshold,

            cooldown=self.config.cooldown,

            min_lr=self.config.eta_min,

        )

    # ----------------------------------------------------------
    # One Cycle
    # ----------------------------------------------------------

    def build_onecycle(self):

        return OneCycleLR(

            optimizer=self.optimizer,

            max_lr=self.config.max_lr,

            epochs=self.config.epochs,

            steps_per_epoch=self.config.steps_per_epoch,

            pct_start=0.30,

            anneal_strategy="cos",

            div_factor=25,

            final_div_factor=10000,

        )

    # ----------------------------------------------------------
    # Warmup + Cosine
    # ----------------------------------------------------------

    def build_warmup_cosine(self):

        warmup = LinearLR(

            optimizer=self.optimizer,

            start_factor=0.01,

            end_factor=1.0,

            total_iters=self.config.warmup_epochs,

        )

        cosine = CosineAnnealingLR(

            optimizer=self.optimizer,

            T_max=self.config.epochs - self.config.warmup_epochs,

            eta_min=self.config.eta_min,

        )

        return SequentialLR(

            optimizer=self.optimizer,

            schedulers=[

                warmup,

                cosine,

            ],

            milestones=[

                self.config.warmup_epochs,

            ],

        )

    # ----------------------------------------------------------
    # Scheduler Step
    # ----------------------------------------------------------

    @staticmethod
    def step(

        scheduler,

        loss=None,

    ):

        if scheduler is None:

            return

        if isinstance(

            scheduler,

            ReduceLROnPlateau,

        ):

            scheduler.step(loss)

        else:

            scheduler.step()

    # ----------------------------------------------------------
    # Current LR
    # ----------------------------------------------------------

    @staticmethod
    def get_lr(

        optimizer,

    ):

        return optimizer.param_groups[0]["lr"]
    # ============================================================
# Save Scheduler State
# ============================================================

def save_scheduler(
    scheduler,
    path,
):
    """
    Save scheduler state dictionary.
    """

    if scheduler is None:
        return

    torch.save(
        scheduler.state_dict(),
        path,
    )


# ============================================================
# Load Scheduler State
# ============================================================

def load_scheduler(
    scheduler,
    path,
):
    """
    Load scheduler state dictionary.
    """

    if scheduler is None:
        return

    state = torch.load(
        path,
        map_location="cpu",
    )

    scheduler.load_state_dict(
        state
    )


# ============================================================
# Print Learning Rate
# ============================================================

def print_lr(
    optimizer,
):

    print("=" * 60)

    print("Learning Rates")

    print("=" * 60)

    for idx, group in enumerate(
        optimizer.param_groups
    ):

        print(
            f"Group {idx+1:2d} : {group['lr']:.8f}"
        )

    print("=" * 60)


# ============================================================
# Learning Rate History
# ============================================================

def simulate_scheduler(
    scheduler,
    optimizer,
    epochs=20,
):

    history = []

    for epoch in range(epochs):

        lr = optimizer.param_groups[0]["lr"]

        history.append(lr)

        if isinstance(
            scheduler,
            ReduceLROnPlateau,
        ):

            scheduler.step(1.0)

        else:

            scheduler.step()

    return history


# ============================================================
# Self Test
# ============================================================

def self_test():

    import torch.nn as nn

    print("=" * 70)
    print("Scheduler Self Test")
    print("=" * 70)

    model = nn.Linear(10, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    config = SchedulerConfig(
        scheduler=SchedulerType.WARMUP_COSINE,
        epochs=20,
        warmup_epochs=3,
        eta_min=1e-6,
    )

    builder = SchedulerBuilder(
        optimizer,
        config,
    )

    scheduler = builder.build()

    print()

    print("Scheduler Type")

    print(type(scheduler))

    print()

    history = simulate_scheduler(

        scheduler,

        optimizer,

        epochs=20,

    )

    print("LR History")

    print(history)

    print()

    print_lr(optimizer)

    print()

    print("Scheduler OK")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    self_test()        
