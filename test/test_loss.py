import torch

from losses.multitask_loss import MultiTaskLoss


criterion = MultiTaskLoss()

outputs = {

    "histology_logits": torch.randn(4,4),

    "stage_logits": torch.randn(4,4),

    "survival_prediction": torch.randn(4,1),

}

targets = {

    "histology": torch.randint(0,4,(4,)),

    "stage": torch.randint(0,4,(4,)),

    "survival": torch.randn(4),

}

loss = criterion(outputs, targets)

print(loss)