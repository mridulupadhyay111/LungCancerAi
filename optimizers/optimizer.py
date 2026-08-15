"""
optimizer.py

Research Grade Optimizer Builder

Supports
--------
✓ Adam
✓ AdamW
✓ SGD
✓ RMSprop
✓ Weight Decay
✓ Parameter Groups

Author
------
LungCancerAI
"""

from dataclasses import dataclass
from enum import Enum

import torch


# ==========================================================
# Optimizer Types
# ==========================================================

class OptimizerType(Enum):

    ADAM = "adam"

    ADAMW = "adamw"

    SGD = "sgd"

    RMSPROP = "rmsprop"


# ==========================================================
# Config
# ==========================================================

@dataclass
class OptimizerConfig:

    optimizer: OptimizerType = OptimizerType.ADAMW

    lr: float = 1e-4

    weight_decay: float = 1e-4

    momentum: float = 0.9

    betas = (0.9, 0.999)


# ==========================================================
# Builder
# ==========================================================

def build_optimizer(
    model,
    config: OptimizerConfig,
):

    params = filter(
        lambda p: p.requires_grad,
        model.parameters(),
    )

    if config.optimizer == OptimizerType.ADAM:

        optimizer = torch.optim.Adam(

            params,

            lr=config.lr,

            weight_decay=config.weight_decay,

            betas=config.betas,

        )

    elif config.optimizer == OptimizerType.ADAMW:

        optimizer = torch.optim.AdamW(

            params,

            lr=config.lr,

            weight_decay=config.weight_decay,

            betas=config.betas,

        )

    elif config.optimizer == OptimizerType.SGD:

        optimizer = torch.optim.SGD(

            params,

            lr=config.lr,

            momentum=config.momentum,

            weight_decay=config.weight_decay,

        )

    elif config.optimizer == OptimizerType.RMSPROP:

        optimizer = torch.optim.RMSprop(

            params,

            lr=config.lr,

            momentum=config.momentum,

            weight_decay=config.weight_decay,

        )

    else:

        raise ValueError("Unsupported optimizer.")

    return optimizer


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(20, 4)

    config = OptimizerConfig()

    optimizer = build_optimizer(

        model,

        config,

    )

    print(optimizer)