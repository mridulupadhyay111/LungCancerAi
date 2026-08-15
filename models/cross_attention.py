"""
cross_attention.py

Research-grade Cross Attention Fusion Module

Author
------
LungCancerAI

Description
-----------
Fuses CT image embeddings and clinical embeddings
using Transformer-style Multi-Head Cross Attention.

Image Features  ---> Query
Clinical -------> Key
Clinical -------> Value

Output
------
Fused embedding for downstream multitask prediction.
"""

from __future__ import annotations

from typing import Optional
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross Attention Fusion Module

    Parameters
    ----------
    image_dim : int

    clinical_dim : int

    hidden_dim : int

    num_heads : int

    dropout : float
    """

    def __init__(

        self,

        image_dim: int = 512,

        clinical_dim: int = 256,

        hidden_dim: int = 512,

        num_heads: int = 8,

        dropout: float = 0.1,

    ):

        super().__init__()

        self.image_dim = image_dim

        self.clinical_dim = clinical_dim

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.dropout_rate = dropout

        # -------------------------------------------------
        # Projection Layers
        # -------------------------------------------------

        self.image_projection = nn.Linear(

            image_dim,

            hidden_dim,

        )

        self.clinical_projection = nn.Linear(

            clinical_dim,

            hidden_dim,

        )

        # -------------------------------------------------
        # Multihead Attention
        # -------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(

            embed_dim=hidden_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True,

        )

        # -------------------------------------------------
        # Residual LayerNorm
        # -------------------------------------------------

        self.norm1 = nn.LayerNorm(

            hidden_dim,

        )

        self.norm2 = nn.LayerNorm(

            hidden_dim,

        )

        # -------------------------------------------------
        # Feed Forward Network
        # -------------------------------------------------

        self.ffn = nn.Sequential(

            nn.Linear(

                hidden_dim,

                hidden_dim * 4,

            ),

            nn.GELU(),

            nn.Dropout(

                dropout,

            ),

            nn.Linear(

                hidden_dim * 4,

                hidden_dim,

            ),

            nn.Dropout(

                dropout,

            ),

        )

        self.dropout = nn.Dropout(

            dropout,

        )

        self._initialize_weights()

    # =====================================================
    # Initialize
    # =====================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(

                module,

                nn.Linear,

            ):

                nn.init.xavier_uniform_(

                    module.weight,

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias,

                    )
        # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        image_embedding: torch.Tensor,
        clinical_embedding: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        image_embedding : (B, image_dim)

        clinical_embedding : (B, clinical_dim)

        return_attention : bool

        Returns
        -------
        fused_embedding

        OR

        {
            fused_embedding,
            attention_weights
        }
        """

        # --------------------------------------------
        # Validate
        # --------------------------------------------

        if image_embedding.ndim != 2:

            raise ValueError(
                "Image embedding must be (B,D)"
            )

        if clinical_embedding.ndim != 2:

            raise ValueError(
                "Clinical embedding must be (B,D)"
            )

        # --------------------------------------------
        # Project into common space
        # --------------------------------------------

        image = self.image_projection(
            image_embedding
        )

        clinical = self.clinical_projection(
            clinical_embedding
        )

        # --------------------------------------------
        # Convert to sequence length = 1
        # MultiheadAttention expects:
        #
        # (Batch, Sequence, Embedding)
        # --------------------------------------------

        query = image.unsqueeze(1)

        key = clinical.unsqueeze(1)

        value = clinical.unsqueeze(1)

        # --------------------------------------------
        # Cross Attention
        # --------------------------------------------

        attended, attention_weights = self.cross_attention(

            query=query,

            key=key,

            value=value,

            need_weights=True,

            average_attn_weights=False,

        )

        # --------------------------------------------
        # First Residual Connection
        # --------------------------------------------

        x = self.norm1(

            query +

            self.dropout(attended)

        )

        # --------------------------------------------
        # Feed Forward
        # --------------------------------------------

        ff = self.ffn(x)

        # --------------------------------------------
        # Second Residual
        # --------------------------------------------

        x = self.norm2(

            x +

            ff

        )

        fused = x.squeeze(1)

        fused = F.normalize(

            fused,

            dim=1,

        )

        if return_attention:

            return {

                "fused_embedding": fused,

                "attention_weights": attention_weights,

                "image_embedding": image,

                "clinical_embedding": clinical,

            }

        return fused

    # =====================================================
    # Extract Attention
    # =====================================================

    @torch.no_grad()

    def attention_map(

        self,

        image_embedding,

        clinical_embedding,

    ):

        self.eval()

        outputs = self.forward(

            image_embedding,

            clinical_embedding,

            return_attention=True,

        )

        return outputs["attention_weights"]

    # =====================================================
    # Cosine Similarity
    # =====================================================

    @staticmethod

    def similarity(

        embedding1,

        embedding2,

    ):

        embedding1 = F.normalize(

            embedding1,

            dim=1,

        )

        embedding2 = F.normalize(

            embedding2,

            dim=1,

        )

        similarity = torch.sum(

            embedding1 *

            embedding2,

            dim=1,

        )

        return similarity

    # =====================================================
    # Embedding Distance
    # =====================================================

    @staticmethod

    def distance(

        embedding1,

        embedding2,

    ):

        return torch.norm(

            embedding1 -

            embedding2,

            dim=1,

        )

    # =====================================================
    # Output Dimension
    # =====================================================

    @property

    def output_dim(self):

        return self.hidden_dim
        # =====================================================
    # Freeze Module
    # =====================================================

    def freeze(self):
        """
        Freeze all parameters.
        """

        for parameter in self.parameters():

            parameter.requires_grad = False

    # =====================================================
    # Unfreeze Module
    # =====================================================

    def unfreeze(self):
        """
        Unfreeze all parameters.
        """

        for parameter in self.parameters():

            parameter.requires_grad = True

    # =====================================================
    # Save Weights
    # =====================================================

    def save_weights(
        self,
        save_path: str,
    ):

        torch.save(
            {
                "state_dict": self.state_dict()
            },
            save_path,
        )

        print(
            f"Checkpoint saved to {save_path}"
        )

    # =====================================================
    # Load Weights
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

        print("Cross Attention Weights Loaded")

        print("=" * 70)

        print("Missing Keys    :", len(missing))

        print("Unexpected Keys :", len(unexpected))

        print("=" * 70)

    # =====================================================
    # Number of Parameters
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

        total = self.num_parameters(False)

        trainable = self.num_parameters(True)

        return {

            "image_dim": self.image_dim,

            "clinical_dim": self.clinical_dim,

            "hidden_dim": self.hidden_dim,

            "num_heads": self.num_heads,

            "dropout": self.dropout_rate,

            "total_parameters": total,

            "trainable_parameters": trainable,

        }

    # =====================================================
    # Print Summary
    # =====================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("Cross Attention Summary")

        print("=" * 70)

        for key, value in stats.items():

            print(

                f"{key:25s}: {value}"

            )

        print("=" * 70)

    # =====================================================
    # Inference
    # =====================================================

    @torch.no_grad()

    def infer(
        self,
        image_embedding,
        clinical_embedding,
    ):

        self.eval()

        return self.forward(

            image_embedding,

            clinical_embedding,

        )

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

            f"TorchScript exported to {save_path}"

        )

    # =====================================================
    # ONNX Export
    # =====================================================

    def export_onnx(

        self,

        save_path: str,

        image_shape=(1, 512),

        clinical_shape=(1, 256),

    ):

        self.eval()

        image = torch.randn(*image_shape)

        clinical = torch.randn(*clinical_shape)

        torch.onnx.export(

            self,

            (image, clinical),

            save_path,

            export_params=True,

            opset_version=17,

            do_constant_folding=True,

            input_names=[

                "image_embedding",

                "clinical_embedding",

            ],

            output_names=[

                "fused_embedding",

            ],

            dynamic_axes={

                "image_embedding": {

                    0: "batch"

                },

                "clinical_embedding": {

                    0: "batch"

                },

                "fused_embedding": {

                    0: "batch"

                },

            },

        )

        print(

            f"ONNX exported to {save_path}"

        )
        # =====================================================
    # Validate Inputs
    # =====================================================

    def validate_inputs(
        self,
        image_embedding: torch.Tensor,
        clinical_embedding: torch.Tensor,
    ):

        if not isinstance(image_embedding, torch.Tensor):
            raise TypeError(
                "image_embedding must be torch.Tensor"
            )

        if not isinstance(clinical_embedding, torch.Tensor):
            raise TypeError(
                "clinical_embedding must be torch.Tensor"
            )

        if image_embedding.ndim != 2:
            raise ValueError(
                f"Expected image embedding shape (B,{self.image_dim})"
            )

        if clinical_embedding.ndim != 2:
            raise ValueError(
                f"Expected clinical embedding shape (B,{self.clinical_dim})"
            )

        if image_embedding.shape[0] != clinical_embedding.shape[0]:
            raise ValueError(
                "Batch size mismatch."
            )

        if image_embedding.shape[1] != self.image_dim:
            raise ValueError(
                f"Expected image_dim={self.image_dim}, "
                f"received {image_embedding.shape[1]}"
            )

        if clinical_embedding.shape[1] != self.clinical_dim:
            raise ValueError(
                f"Expected clinical_dim={self.clinical_dim}, "
                f"received {clinical_embedding.shape[1]}"
            )

        if torch.isnan(image_embedding).any():
            raise ValueError(
                "NaN values detected in image embedding."
            )

        if torch.isnan(clinical_embedding).any():
            raise ValueError(
                "NaN values detected in clinical embedding."
            )

        if torch.isinf(image_embedding).any():
            raise ValueError(
                "Infinite values detected in image embedding."
            )

        if torch.isinf(clinical_embedding).any():
            raise ValueError(
                "Infinite values detected in clinical embedding."
            )

        return True

    # =====================================================
    # Batch Inference
    # =====================================================

    @torch.no_grad()
    def fuse_batch(
        self,
        image_embedding,
        clinical_embedding,
    ):

        self.eval()

        self.validate_inputs(
            image_embedding,
            clinical_embedding,
        )

        return self.forward(
            image_embedding,
            clinical_embedding,
        )

    # =====================================================
    # Get Attention Matrix
    # =====================================================

    @torch.no_grad()
    def get_attention_matrix(
        self,
        image_embedding,
        clinical_embedding,
    ):

        outputs = self.forward(
            image_embedding,
            clinical_embedding,
            return_attention=True,
        )

        return outputs["attention_weights"]

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
    # Device
    # =====================================================

    @property
    def device(self):

        return next(
            self.parameters()
        ).device

    # =====================================================
    # Enable Gradient Checkpointing
    # =====================================================

    def enable_gradient_checkpointing(self):

        self.gradient_checkpointing = True

    # =====================================================
    # Disable Gradient Checkpointing
    # =====================================================

    def disable_gradient_checkpointing(self):

        self.gradient_checkpointing = False

    # =====================================================
    # Gradient Norm
    # =====================================================

    def gradient_norm(self):

        total_norm = 0.0

        for parameter in self.parameters():

            if parameter.grad is None:
                continue

            norm = parameter.grad.data.norm(2)

            total_norm += norm.item() ** 2

        return total_norm ** 0.5

    # =====================================================
    # Reset Gradients
    # =====================================================

    def reset_gradients(self):

        for parameter in self.parameters():

            parameter.grad = None

    # =====================================================
    # Mixed Precision Compatibility
    # =====================================================

    def cast_inputs(
        self,
        image_embedding,
        clinical_embedding,
    ):

        dtype = next(
            self.parameters()
        ).dtype

        image_embedding = image_embedding.to(dtype)

        clinical_embedding = clinical_embedding.to(dtype)

        return image_embedding, clinical_embedding
    # =====================================================
# Factory Function
# =====================================================

def build_cross_attention(
    image_dim: int = 512,
    clinical_dim: int = 256,
    hidden_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.10,
):

    """
    Factory function.

    Example
    -------
    fusion = build_cross_attention()
    """

    return CrossAttention(

        image_dim=image_dim,

        clinical_dim=clinical_dim,

        hidden_dim=hidden_dim,

        num_heads=num_heads,

        dropout=dropout,

    )


# =====================================================
# Dummy Inputs
# =====================================================

def dummy_inputs(
    batch_size: int = 4,
    image_dim: int = 512,
    clinical_dim: int = 256,
):

    image = torch.randn(
        batch_size,
        image_dim,
    )

    clinical = torch.randn(
        batch_size,
        clinical_dim,
    )

    return image, clinical


# =====================================================
# Compare Fused Embeddings
# =====================================================

def compare_embeddings(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
):

    cosine = F.cosine_similarity(
        embedding1,
        embedding2,
        dim=1,
    )

    euclidean = torch.norm(
        embedding1 - embedding2,
        dim=1,
    )

    return {

        "cosine_similarity": cosine,

        "euclidean_distance": euclidean,

    }


# =====================================================
# Self Test
# =====================================================

def self_test():

    print("=" * 70)
    print("Cross Attention Self Test")
    print("=" * 70)

    fusion = build_cross_attention()

    fusion.print_summary()

    image, clinical = dummy_inputs()

    with torch.no_grad():

        outputs = fusion(
            image,
            clinical,
            return_attention=True,
        )

    fused = outputs["fused_embedding"]

    attention = outputs["attention_weights"]

    print()

    print("Image Embedding Shape")

    print(image.shape)

    print()

    print("Clinical Embedding Shape")

    print(clinical.shape)

    print()

    print("Fused Embedding Shape")

    print(fused.shape)

    print()

    print("Attention Shape")

    print(attention.shape)

    assert fused.shape == (

        image.shape[0],

        fusion.hidden_dim,

    )

    print()

    print("Self Test Passed")

    print("=" * 70)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    self_test()                            