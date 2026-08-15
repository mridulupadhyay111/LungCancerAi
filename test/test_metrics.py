import torch

from metrics.evaluator import Evaluator

evaluator = Evaluator()

outputs = {

    "histology_logits": torch.randn(8,5),

    "stage_logits": torch.randn(8,4),

    "survival_prediction": torch.randn(8,1),

}

targets = {

    "histology": torch.randint(0,5,(8,)),

    "stage": torch.randint(0,4,(8,)),

    "survival": torch.rand(8)*1000,

}

result = evaluator.evaluate(

    outputs,

    targets,

)

print(result)