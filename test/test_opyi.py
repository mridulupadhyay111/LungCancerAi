from optimizers.optimizer import *

import torch.nn as nn

model = nn.Linear(10,4)

optimizer = build_optimizer(
    model,
    OptimizerConfig()
)

print(optimizer)