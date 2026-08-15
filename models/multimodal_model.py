"""
multimodal_model.py

Research-grade Multimodal Lung Cancer AI

Author
------
LungCancerAI

Architecture
------------
CT Image
    │
Image Encoder
    │
Image Embedding (512)

Clinical Features
    │
Clinical Encoder
    │
Clinical Embedding (256)

        │
        ▼
Cross Attention Fusion
        │
Fused Embedding (512)
        │
        ▼
MultiTask Heads

Outputs
-------
Histology Classification
Stage Prediction
Survival Prediction
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import importlib.util
from pathlib import Path

from models.imageencoder import ImageEncoder
from models.cross_attention import CrossAttention
from models.multitask_heads import (
    HeadConfig,
    MultiTaskHeads,
)

_clinical_encoder_path = Path(__file__).with_name("clinical_encoder.py")
_spec = importlib.util.spec_from_file_location("clinical_encoder_module", _clinical_encoder_path)
_clinical_encoder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_clinical_encoder_module)
ClinicalEncoder = _clinical_encoder_module.ClinicalEncoder


class MultiModalModel(nn.Module):
    """
    Complete Multimodal Network

    Components
    ----------
    1. Image Encoder
    2. Clinical Encoder
    3. Cross Attention
    4. Multi-task Prediction Heads
    """

    def __init__(
        self,
        clinical_input_dim: int = 20,
        image_embedding_dim: int = 512,
        clinical_embedding_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.30,
        num_histology_classes: int = 5,
        num_stage_classes: int = 4,
    ):

        super().__init__()

        ####################################################
        # Image Encoder
        ####################################################

        self.image_encoder = ImageEncoder(

            embedding_dim=image_embedding_dim,

            dropout=dropout,

        )

        ####################################################
        # Clinical Encoder
        ####################################################

        self.clinical_encoder = ClinicalEncoder(

            input_dim=clinical_input_dim,

            embedding_dim=clinical_embedding_dim,

            dropout=dropout,

        )

        ####################################################
        # Cross Attention
        ####################################################

        self.cross_attention = CrossAttention(

            image_dim=image_embedding_dim,

            clinical_dim=clinical_embedding_dim,

            hidden_dim=fusion_dim,

            num_heads=num_heads,

            dropout=dropout,

        )

        ####################################################
        # Prediction Heads
        ####################################################

        head_config = HeadConfig(

            embedding_dim=fusion_dim,

            hidden_dim=256,

            dropout=dropout,

            num_histology_classes=num_histology_classes,

            num_stage_classes=num_stage_classes,

        )

        self.prediction_heads = MultiTaskHeads(

            head_config

        )

        self.image_embedding_dim = image_embedding_dim

        self.clinical_embedding_dim = clinical_embedding_dim

        self.fusion_dim = fusion_dim
        # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        image
            Shape (B,1,128,128,128)

        clinical
            Shape (B,num_features)

        Returns
        -------
        Dictionary containing predictions and
        optionally intermediate features.
        """

        # --------------------------------------------
        # Image Encoder
        # --------------------------------------------

        image_embedding = self.image_encoder(

            image

        )

        # --------------------------------------------
        # Clinical Encoder
        # --------------------------------------------

        clinical_embedding = self.clinical_encoder(

            clinical

        )

        # --------------------------------------------
        # Cross Attention Fusion
        # --------------------------------------------

        if return_attention:

            fusion_outputs = self.cross_attention(

                image_embedding,

                clinical_embedding,

                return_attention=True,

            )

            fused_embedding = fusion_outputs[

                "fused_embedding"

            ]

            attention_weights = fusion_outputs[

                "attention_weights"

            ]

        else:

            fused_embedding = self.cross_attention(

                image_embedding,

                clinical_embedding,

            )

            attention_weights = None

        # --------------------------------------------
        # Multi-task Heads
        # --------------------------------------------

        predictions = self.prediction_heads(

            fused_embedding,

            return_probabilities=True,

        )

        # --------------------------------------------
        # Optional Outputs
        # --------------------------------------------

        outputs = {

            **predictions

        }

        if return_features:

            outputs.update(

                {

                    "image_embedding":

                        image_embedding,

                    "clinical_embedding":

                        clinical_embedding,

                    "fused_embedding":

                        fused_embedding,

                }

            )

        if return_attention:

            outputs[

                "attention_weights"

            ] = attention_weights

        return outputs

    # =====================================================
    # Encode Image Only
    # =====================================================

    @torch.no_grad()

    def encode_image(

        self,

        image,

    ):

        self.eval()

        return self.image_encoder(

            image

        )

    # =====================================================
    # Encode Clinical Only
    # =====================================================

    @torch.no_grad()

    def encode_clinical(

        self,

        clinical,

    ):

        self.eval()

        return self.clinical_encoder(

            clinical

        )

    # =====================================================
    # Fuse Embeddings
    # =====================================================

    @torch.no_grad()

    def fuse(

        self,

        image_embedding,

        clinical_embedding,

    ):

        self.eval()

        return self.cross_attention(

            image_embedding,

            clinical_embedding,

        )
        # =====================================================
    # Prediction API
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ):

        self.eval()

        return self.forward(

            image,

            clinical,

            return_features=False,

            return_attention=False,

        )

    # =====================================================
    # Freeze Modules
    # =====================================================

    def freeze_image_encoder(self):

        self.image_encoder.freeze_backbone()

        self.image_encoder.freeze_projection()

    def freeze_clinical_encoder(self):

        self.clinical_encoder.freeze()

    def freeze_cross_attention(self):

        self.cross_attention.freeze()

    def freeze_prediction_heads(self):

        self.prediction_heads.freeze()

    # =====================================================
    # Unfreeze Modules
    # =====================================================

    def unfreeze_image_encoder(self):

        self.image_encoder.unfreeze_backbone()

        self.image_encoder.unfreeze_projection()

    def unfreeze_clinical_encoder(self):

        self.clinical_encoder.unfreeze()

    def unfreeze_cross_attention(self):

        self.cross_attention.unfreeze()

    def unfreeze_prediction_heads(self):

        self.prediction_heads.unfreeze()

    # =====================================================
    # Freeze Entire Model
    # =====================================================

    def freeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = False

    # =====================================================
    # Unfreeze Entire Model
    # =====================================================

    def unfreeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = True

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

        print("Multimodal Model Loaded")

        print("=" * 70)

        print("Missing Keys    :", len(missing))

        print("Unexpected Keys :", len(unexpected))

        print("=" * 70)

    # =====================================================
    # Parameter Count
    # =====================================================

    def num_parameters(

        self,

        trainable_only=True,

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
    # Statistics
    # =====================================================

    def statistics(self):

        return {

            "image_embedding_dim":

                self.image_embedding_dim,

            "clinical_embedding_dim":

                self.clinical_embedding_dim,

            "fusion_dim":

                self.fusion_dim,

            "total_parameters":

                self.num_parameters(False),

            "trainable_parameters":

                self.num_parameters(True),

        }

    # =====================================================
    # Summary
    # =====================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("Multimodal LungCancerAI Model")

        print("=" * 70)

        for key, value in stats.items():

            print(

                f"{key:30s}: {value}"

            )

        print("=" * 70)
        # =====================================================
    # Prediction API
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ):

        self.eval()

        return self.forward(

            image,

            clinical,

            return_features=False,

            return_attention=False,

        )

    # =====================================================
    # Freeze Modules
    # =====================================================

    def freeze_image_encoder(self):

        self.image_encoder.freeze_backbone()

        self.image_encoder.freeze_projection()

    def freeze_clinical_encoder(self):

        self.clinical_encoder.freeze()

    def freeze_cross_attention(self):

        self.cross_attention.freeze()

    def freeze_prediction_heads(self):

        self.prediction_heads.freeze()

    # =====================================================
    # Unfreeze Modules
    # =====================================================

    def unfreeze_image_encoder(self):

        self.image_encoder.unfreeze_backbone()

        self.image_encoder.unfreeze_projection()

    def unfreeze_clinical_encoder(self):

        self.clinical_encoder.unfreeze()

    def unfreeze_cross_attention(self):

        self.cross_attention.unfreeze()

    def unfreeze_prediction_heads(self):

        self.prediction_heads.unfreeze()

    # =====================================================
    # Freeze Entire Model
    # =====================================================

    def freeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = False

    # =====================================================
    # Unfreeze Entire Model
    # =====================================================

    def unfreeze(self):

        for parameter in self.parameters():

            parameter.requires_grad = True

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

        print("Multimodal Model Loaded")

        print("=" * 70)

        print("Missing Keys    :", len(missing))

        print("Unexpected Keys :", len(unexpected))

        print("=" * 70)

    # =====================================================
    # Parameter Count
    # =====================================================

    def num_parameters(

        self,

        trainable_only=True,

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
    # Statistics
    # =====================================================

    def statistics(self):

        return {

            "image_embedding_dim":

                self.image_embedding_dim,

            "clinical_embedding_dim":

                self.clinical_embedding_dim,

            "fusion_dim":

                self.fusion_dim,

            "total_parameters":

                self.num_parameters(False),

            "trainable_parameters":

                self.num_parameters(True),

        }

    # =====================================================
    # Summary
    # =====================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("Multimodal LungCancerAI Model")

        print("=" * 70)

        for key, value in stats.items():

            print(

                f"{key:30s}: {value}"

            )

        print("=" * 70)
        # =====================================================
    # Feature Extraction
    # =====================================================

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ):
        """
        Returns intermediate embeddings for analysis,
        visualization and downstream tasks.
        """

        self.eval()

        image_embedding = self.image_encoder(image)

        clinical_embedding = self.clinical_encoder(clinical)

        fusion_outputs = self.cross_attention(
            image_embedding,
            clinical_embedding,
            return_attention=True,
        )

        return {

            "image_embedding":
                image_embedding,

            "clinical_embedding":
                clinical_embedding,

            "fused_embedding":
                fusion_outputs["fused_embedding"],

            "attention_weights":
                fusion_outputs["attention_weights"],

        }

    # =====================================================
    # Extract Image Embedding
    # =====================================================

    @torch.no_grad()
    def extract_image_embedding(
        self,
        image: torch.Tensor,
    ):

        self.eval()

        return self.image_encoder(image)

    # =====================================================
    # Extract Clinical Embedding
    # =====================================================

    @torch.no_grad()
    def extract_clinical_embedding(
        self,
        clinical: torch.Tensor,
    ):

        self.eval()

        return self.clinical_encoder(clinical)

    # =====================================================
    # Extract Fused Embedding
    # =====================================================

    @torch.no_grad()
    def extract_fused_embedding(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ):

        self.eval()

        image_embedding = self.image_encoder(image)

        clinical_embedding = self.clinical_encoder(clinical)

        fused_embedding = self.cross_attention(
            image_embedding,
            clinical_embedding,
        )

        return fused_embedding

    # =====================================================
    # Extract Attention
    # =====================================================

    @torch.no_grad()
    def extract_attention(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ):

        self.eval()

        image_embedding = self.image_encoder(image)

        clinical_embedding = self.clinical_encoder(clinical)

        outputs = self.cross_attention(
            image_embedding,
            clinical_embedding,
            return_attention=True,
        )

        return outputs["attention_weights"]

    # =====================================================
    # Register Forward Hook
    # =====================================================

    def register_feature_hook(
        self,
        module_name: str,
        hook_fn,
    ):
        """
        Register a forward hook for Grad-CAM,
        activation visualization, etc.
        """

        modules = {

            "image_encoder":
                self.image_encoder,

            "clinical_encoder":
                self.clinical_encoder,

            "cross_attention":
                self.cross_attention,

            "prediction_heads":
                self.prediction_heads,

        }

        if module_name not in modules:

            raise ValueError(
                f"Unknown module: {module_name}"
            )

        return modules[module_name].register_forward_hook(
            hook_fn
        )

    # =====================================================
    # Register Backward Hook
    # =====================================================

    def register_backward_hook(
        self,
        module_name: str,
        hook_fn,
    ):

        modules = {

            "image_encoder":
                self.image_encoder,

            "clinical_encoder":
                self.clinical_encoder,

            "cross_attention":
                self.cross_attention,

            "prediction_heads":
                self.prediction_heads,

        }

        if module_name not in modules:

            raise ValueError(
                f"Unknown module: {module_name}"
            )

        return modules[module_name].register_full_backward_hook(
            hook_fn
        )

    # =====================================================
    # Enable Explainability Mode
    # =====================================================

    def enable_xai(self):
        """
        Enable explainability mode.
        Future Grad-CAM and attention visualization
        modules can use this flag.
        """

        self.xai_enabled = True

    # =====================================================
    # Disable Explainability Mode
    # =====================================================

    def disable_xai(self):

        self.xai_enabled = False

    # =====================================================
    # Is Explainability Enabled
    # =====================================================

    @property
    def explainability_enabled(self):

        return getattr(
            self,
            "xai_enabled",
            False,
        )
    # =====================================================
# Factory Function
# =====================================================

def build_multimodal_model(
    clinical_input_dim: int = 20,
    image_embedding_dim: int = 512,
    clinical_embedding_dim: int = 256,
    fusion_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.30,
    num_histology_classes: int = 5,
    num_stage_classes: int = 4,
):

    return MultiModalModel(

        clinical_input_dim=clinical_input_dim,

        image_embedding_dim=image_embedding_dim,

        clinical_embedding_dim=clinical_embedding_dim,

        fusion_dim=fusion_dim,

        num_heads=num_heads,

        dropout=dropout,

        num_histology_classes=num_histology_classes,

        num_stage_classes=num_stage_classes,

    )


# =====================================================
# Dummy Inputs
# =====================================================

def dummy_inputs(

    batch_size=2,

    clinical_features=20,

):

    image = torch.randn(

        batch_size,

        1,

        128,

        128,

        128,

    )

    clinical = torch.randn(

        batch_size,

        clinical_features,

    )

    return image, clinical


# =====================================================
# Prediction Summary
# =====================================================

def summarize_predictions(outputs):

    print("=" * 70)

    print("Prediction Summary")

    print("=" * 70)

    print(

        "Histology:",

        outputs["histology_logits"].shape,

    )

    print(

        "Stage:",

        outputs["stage_logits"].shape,

    )

    print(

        "Survival:",

        outputs["survival_prediction"].shape,

    )

    if "histology_prediction" in outputs:

        print(

            "Predicted Histology:",

            outputs["histology_prediction"],

        )

    if "stage_prediction" in outputs:

        print(

            "Predicted Stage:",

            outputs["stage_prediction"],

        )

    print("=" * 70)


# =====================================================
# Self Test
# =====================================================

def self_test():

    print("=" * 70)

    print("Running MultiModalModel Self Test")

    print("=" * 70)

    model = build_multimodal_model(

        clinical_input_dim=20,

    )

    model.print_summary()

    image, clinical = dummy_inputs(

        batch_size=4,

        clinical_features=20,

    )

    with torch.no_grad():

        outputs = model.predict(

            image,

            clinical,

        )

    summarize_predictions(outputs)

    assert outputs["histology_logits"].shape == (4, 5)

    assert outputs["stage_logits"].shape == (4, 4)

    assert outputs["survival_prediction"].shape == (4, 1)

    print()

    print("Extracting Intermediate Features...")

    features = model.extract_features(

        image,

        clinical,

    )

    print(

        "Image Embedding:",

        features["image_embedding"].shape,

    )

    print(

        "Clinical Embedding:",

        features["clinical_embedding"].shape,

    )

    print(

        "Fused Embedding:",

        features["fused_embedding"].shape,

    )

    print(

        "Attention:",

        features["attention_weights"].shape,

    )

    print()

    print("Self Test Passed")

    print("=" * 70)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    self_test()                    