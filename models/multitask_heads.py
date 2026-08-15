"""
multitask_heads.py

Research-grade Multi-Task Prediction Heads

Tasks
-----
1. Histology Classification
2. Stage Prediction
3. Survival Prediction

Author
------
LungCancerAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class HeadConfig:

    embedding_dim: int = 512

    hidden_dim: int = 256

    dropout: float = 0.30

    num_histology_classes: int = 5

    num_stage_classes: int = 4


# ==========================================================
# Base Prediction Head
# ==========================================================

class PredictionHead(nn.Module):

    def __init__(

        self,

        input_dim,

        hidden_dim,

        output_dim,

        dropout,

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.BatchNorm1d(
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.BatchNorm1d(
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                output_dim,
            )

        )

        self.initialize()

    # -----------------------------------------------------

    def initialize(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(

                    module.weight,

                    nonlinearity="relu",

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias

                    )

    # -----------------------------------------------------

    def forward(

        self,

        x,

    ):

        return self.network(x)


# ==========================================================
# Histology Head
# ==========================================================

class HistologyHead(PredictionHead):

    """
    Predicts lung cancer subtype.

    Example classes

    0 Adenocarcinoma

    1 Squamous Cell

    2 Large Cell

    3 Other
    """

    def __init__(

        self,

        config: HeadConfig,

    ):

        super().__init__(

            input_dim=config.embedding_dim,

            hidden_dim=config.hidden_dim,

            output_dim=config.num_histology_classes,

            dropout=config.dropout,

        )


# ==========================================================
# Stage Prediction Head
# ==========================================================

class StageHead(PredictionHead):

    """
    Predict

    Stage I

    Stage II

    Stage III

    Stage IV
    """

    def __init__(

        self,

        config: HeadConfig,

    ):

        super().__init__(

            input_dim=config.embedding_dim,

            hidden_dim=config.hidden_dim,

            output_dim=config.num_stage_classes,

            dropout=config.dropout,

        )


# ==========================================================
# Survival Head
# ==========================================================

class SurvivalHead(PredictionHead):

    """
    Survival regression.

    Output

    Survival Time

    (days)
    """

    def __init__(

        self,

        config: HeadConfig,

    ):

        super().__init__(

            input_dim=config.embedding_dim,

            hidden_dim=config.hidden_dim,

            output_dim=1,

            dropout=config.dropout,

        )
    # ==========================================================
# MultiTask Heads
# ==========================================================

class MultiTaskHeads(nn.Module):

    """
    Multi-task prediction module.

    Tasks
    -----
    1. Histology Classification
    2. Stage Classification
    3. Survival Regression
    """

    def __init__(
        self,
        config: HeadConfig,
    ):

        super().__init__()

        self.config = config

        self.histology_head = HistologyHead(config)

        self.stage_head = StageHead(config)

        self.survival_head = SurvivalHead(config)

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        fused_embedding: torch.Tensor,
        return_probabilities: bool = False,
    ):

        if fused_embedding.ndim != 2:

            raise ValueError(
                "Expected fused embedding of shape (B,D)"
            )

        if fused_embedding.shape[1] != self.config.embedding_dim:

            raise ValueError(
                f"Expected embedding dimension "
                f"{self.config.embedding_dim}, "
                f"received {fused_embedding.shape[1]}"
            )

        # -----------------------------------------
        # Forward Pass
        # -----------------------------------------

        histology_logits = self.histology_head(
            fused_embedding
        )

        stage_logits = self.stage_head(
            fused_embedding
        )

        survival_prediction = self.survival_head(
            fused_embedding
        )

        outputs = {

            "histology_logits": histology_logits,

            "stage_logits": stage_logits,

            "survival_prediction": survival_prediction,

        }

        # -----------------------------------------
        # Optional probabilities
        # -----------------------------------------

        if return_probabilities:

            outputs["histology_probabilities"] = F.softmax(

                histology_logits,

                dim=1,

            )

            outputs["stage_probabilities"] = F.softmax(

                stage_logits,

                dim=1,

            )

        return outputs

    # =====================================================
    # Prediction
    # =====================================================

    @torch.no_grad()

    def predict(
        self,
        fused_embedding: torch.Tensor,
    ):

        self.eval()

        outputs = self.forward(

            fused_embedding,

            return_probabilities=True,

        )

        outputs["histology_prediction"] = torch.argmax(

            outputs["histology_probabilities"],

            dim=1,

        )

        outputs["stage_prediction"] = torch.argmax(

            outputs["stage_probabilities"],

            dim=1,

        )

        return outputs

    # =====================================================
    # Individual Heads
    # =====================================================

    def predict_histology(
        self,
        fused_embedding,
    ):

        return self.histology_head(

            fused_embedding

        )

    def predict_stage(
        self,
        fused_embedding,
    ):

        return self.stage_head(

            fused_embedding

        )

    def predict_survival(
        self,
        fused_embedding,
    ):

        return self.survival_head(

            fused_embedding

        )

    # =====================================================
    # Embedding Validation
    # =====================================================

    def validate_embedding(
        self,
        embedding,
    ):

        if not isinstance(
            embedding,
            torch.Tensor,
        ):

            raise TypeError(
                "Embedding must be torch.Tensor"
            )

        if torch.isnan(embedding).any():

            raise ValueError(
                "NaN values detected."
            )

        if torch.isinf(embedding).any():

            raise ValueError(
                "Infinite values detected."
            )

        return True
        # =====================================================
    # Freeze All Heads
    # =====================================================

    def freeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = False

    # =====================================================
    # Unfreeze All Heads
    # =====================================================

    def unfreeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = True

    # =====================================================
    # Freeze Individual Heads
    # =====================================================

    def freeze_histology(self):

        for parameter in self.histology_head.parameters():

            parameter.requires_grad = False

    def freeze_stage(self):

        for parameter in self.stage_head.parameters():

            parameter.requires_grad = False

    def freeze_survival(self):

        for parameter in self.survival_head.parameters():

            parameter.requires_grad = False

    # =====================================================
    # Unfreeze Individual Heads
    # =====================================================

    def unfreeze_histology(self):

        for parameter in self.histology_head.parameters():

            parameter.requires_grad = True

    def unfreeze_stage(self):

        for parameter in self.stage_head.parameters():

            parameter.requires_grad = True

    def unfreeze_survival(self):

        for parameter in self.survival_head.parameters():

            parameter.requires_grad = True

    # =====================================================
    # Parameter Count
    # =====================================================

    def num_parameters(
        self,
        trainable_only: bool = True,
    ):

        if trainable_only:

            return sum(

                p.numel()

                for p in self.parameters()

                if p.requires_grad

            )

        return sum(

            p.numel()

            for p in self.parameters()

        )

    # =====================================================
    # Model Statistics
    # =====================================================

    def statistics(self):

        return {

            "embedding_dim":
                self.config.embedding_dim,

            "hidden_dim":
                self.config.hidden_dim,

            "histology_classes":
                self.config.num_histology_classes,

            "stage_classes":
                self.config.num_stage_classes,

            "dropout":
                self.config.dropout,

            "total_parameters":
                self.num_parameters(False),

            "trainable_parameters":
                self.num_parameters(True),

        }

    # =====================================================
    # Print Summary
    # =====================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("MultiTask Prediction Heads")

        print("=" * 70)

        for key, value in stats.items():

            print(

                f"{key:30s}: {value}"

            )

        print("=" * 70)

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_weights(
        self,
        save_path: str,
    ):

        torch.save(

            {

                "state_dict": self.state_dict(),

            },

            save_path,

        )

        print(

            f"Checkpoint saved to {save_path}"

        )

    # =====================================================
    # Load Checkpoint
    # =====================================================

    def load_weights(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ):

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

        )

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:

                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                checkpoint = checkpoint["model_state_dict"]

        cleaned = {}

        for key, value in checkpoint.items():

            if key.startswith("module."):

                key = key[7:]

            cleaned[key] = value

        missing, unexpected = self.load_state_dict(

            cleaned,

            strict=strict,

        )

        print("=" * 70)

        print("MultiTask Heads Loaded")

        print("=" * 70)

        print("Missing Keys    :", len(missing))

        print("Unexpected Keys :", len(unexpected))

        print("=" * 70)

    # =====================================================
    # Inference
    # =====================================================

    @torch.no_grad()

    def infer(
        self,
        fused_embedding,
    ):

        self.eval()

        return self.predict(
            fused_embedding
        )
        # =====================================================
    # Device
    # =====================================================

    @property
    def device(self):

        return next(self.parameters()).device

    # =====================================================
    # Model Size
    # =====================================================

    def model_size_mb(self):

        total_params = sum(

            p.numel()

            for p in self.parameters()

        )

        return total_params * 4 / (1024 ** 2)

    # =====================================================
    # Validate Embedding
    # =====================================================

    def validate(
        self,
        fused_embedding: torch.Tensor,
    ):

        if not isinstance(
            fused_embedding,
            torch.Tensor,
        ):

            raise TypeError(
                "Input must be torch.Tensor"
            )

        if fused_embedding.ndim != 2:

            raise ValueError(
                "Embedding must have shape (B,D)"
            )

        if fused_embedding.shape[1] != self.config.embedding_dim:

            raise ValueError(

                f"Expected embedding dimension "

                f"{self.config.embedding_dim}"

            )

        if torch.isnan(fused_embedding).any():

            raise ValueError(
                "NaN values detected."
            )

        if torch.isinf(fused_embedding).any():

            raise ValueError(
                "Infinite values detected."
            )

        return True

    # =====================================================
    # Batch Prediction
    # =====================================================

    @torch.no_grad()
    def predict_batch(
        self,
        fused_embedding,
    ):

        self.validate(fused_embedding)

        self.eval()

        return self.predict(fused_embedding)

    # =====================================================
    # TorchScript Export
    # =====================================================

    def export_torchscript(
        self,
        save_path: str,
    ):

        self.eval()

        scripted = torch.jit.script(self)

        scripted.save(save_path)

        print(

            f"TorchScript exported to "

            f"{save_path}"

        )

    # =====================================================
    # ONNX Export
    # =====================================================

    def export_onnx(

        self,

        save_path: str,

        input_shape=(1, 512),

    ):

        self.eval()

        dummy = torch.randn(*input_shape)

        torch.onnx.export(

            self,

            dummy,

            save_path,

            export_params=True,

            opset_version=17,

            do_constant_folding=True,

            input_names=[

                "fused_embedding",

            ],

            output_names=[

                "histology_logits",

                "stage_logits",

                "survival_prediction",

            ],

            dynamic_axes={

                "fused_embedding": {

                    0: "batch"

                },

                "histology_logits": {

                    0: "batch"

                },

                "stage_logits": {

                    0: "batch"

                },

                "survival_prediction": {

                    0: "batch"

                },

            },

        )

        print(

            f"ONNX exported to "

            f"{save_path}"

        )

    # =====================================================
    # Enable Training
    # =====================================================

    def enable_training(self):

        self.train()

    # =====================================================
    # Enable Evaluation
    # =====================================================

    def enable_evaluation(self):

        self.eval()

    # =====================================================
    # Gradient Norm
    # =====================================================

    def gradient_norm(self):

        total = 0.0

        for parameter in self.parameters():

            if parameter.grad is None:

                continue

            norm = parameter.grad.data.norm(2)

            total += norm.item() ** 2

        return total ** 0.5

    # =====================================================
    # Reset Gradients
    # =====================================================

    def reset_gradients(self):

        for parameter in self.parameters():

            parameter.grad = None
        # =====================================================
    # Device
    # =====================================================

    @property
    def device(self):

        return next(self.parameters()).device

    # =====================================================
    # Model Size
    # =====================================================

    def model_size_mb(self):

        total_params = sum(

            p.numel()

            for p in self.parameters()

        )

        return total_params * 4 / (1024 ** 2)

    # =====================================================
    # Validate Embedding
    # =====================================================

    def validate(
        self,
        fused_embedding: torch.Tensor,
    ):

        if not isinstance(
            fused_embedding,
            torch.Tensor,
        ):

            raise TypeError(
                "Input must be torch.Tensor"
            )

        if fused_embedding.ndim != 2:

            raise ValueError(
                "Embedding must have shape (B,D)"
            )

        if fused_embedding.shape[1] != self.config.embedding_dim:

            raise ValueError(

                f"Expected embedding dimension "

                f"{self.config.embedding_dim}"

            )

        if torch.isnan(fused_embedding).any():

            raise ValueError(
                "NaN values detected."
            )

        if torch.isinf(fused_embedding).any():

            raise ValueError(
                "Infinite values detected."
            )

        return True

    # =====================================================
    # Batch Prediction
    # =====================================================

    @torch.no_grad()
    def predict_batch(
        self,
        fused_embedding,
    ):

        self.validate(fused_embedding)

        self.eval()

        return self.predict(fused_embedding)

    # =====================================================
    # TorchScript Export
    # =====================================================

    def export_torchscript(
        self,
        save_path: str,
    ):

        self.eval()

        scripted = torch.jit.script(self)

        scripted.save(save_path)

        print(

            f"TorchScript exported to "

            f"{save_path}"

        )

    # =====================================================
    # ONNX Export
    # =====================================================

    def export_onnx(

        self,

        save_path: str,

        input_shape=(1, 512),

    ):

        self.eval()

        dummy = torch.randn(*input_shape)

        torch.onnx.export(

            self,

            dummy,

            save_path,

            export_params=True,

            opset_version=17,

            do_constant_folding=True,

            input_names=[

                "fused_embedding",

            ],

            output_names=[

                "histology_logits",

                "stage_logits",

                "survival_prediction",

            ],

            dynamic_axes={

                "fused_embedding": {

                    0: "batch"

                },

                "histology_logits": {

                    0: "batch"

                },

                "stage_logits": {

                    0: "batch"

                },

                "survival_prediction": {

                    0: "batch"

                },

            },

        )

        print(

            f"ONNX exported to "

            f"{save_path}"

        )

    # =====================================================
    # Enable Training
    # =====================================================

    def enable_training(self):

        self.train()

    # =====================================================
    # Enable Evaluation
    # =====================================================

    def enable_evaluation(self):

        self.eval()

    # =====================================================
    # Gradient Norm
    # =====================================================

    def gradient_norm(self):

        total = 0.0

        for parameter in self.parameters():

            if parameter.grad is None:

                continue

            norm = parameter.grad.data.norm(2)

            total += norm.item() ** 2

        return total ** 0.5

    # =====================================================
    # Reset Gradients
    # =====================================================

    def reset_gradients(self):

        for parameter in self.parameters():

            parameter.grad = None                    