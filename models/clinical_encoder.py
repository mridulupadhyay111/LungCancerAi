"""
clinical_encoder.py

Research-grade Clinical Feature Encoder

Author:
    LungCancerAI

Description
-----------
Encodes structured clinical variables into
a dense embedding suitable for multimodal fusion.

Input
-----
Clinical feature vector

(B, num_features)

Output
------
(B, embedding_dim)

Used together with the CT Image Encoder.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalEncoder(nn.Module):
    """
    Clinical Feature Encoder

    Parameters
    ----------
    input_dim : int
        Number of clinical variables.

    embedding_dim : int
        Output embedding dimension.

    hidden_dims : tuple
        Hidden layer dimensions.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dims=(256, 128),
        dropout: float = 0.30,
    ):

        super().__init__()

        self.input_dim = input_dim

        self.embedding_dim = embedding_dim

        self.hidden_dims = hidden_dims

        self.dropout_rate = dropout

        layers = []

        previous_dim = input_dim

        for hidden in hidden_dims:

            layers.extend(

                [

                    nn.Linear(

                        previous_dim,

                        hidden,

                    ),

                    nn.BatchNorm1d(

                        hidden,

                    ),

                    nn.ReLU(

                        inplace=True,

                    ),

                    nn.Dropout(

                        dropout,

                    ),

                ]

            )

            previous_dim = hidden

        self.feature_extractor = nn.Sequential(

            *layers

        )

        self.projection = nn.Sequential(

            nn.Linear(

                previous_dim,

                embedding_dim,

            ),

            nn.LayerNorm(

                embedding_dim,

            ),

        )

        self._initialize_weights()

    # =====================================================
    # Weight Initialization
    # =====================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(

                module,

                nn.Linear,

            ):

                nn.init.kaiming_normal_(

                    module.weight,

                    nonlinearity="relu",

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias

                    )

    # =====================================================
    # Feature Extraction
    # =====================================================

    def forward_features(

        self,

        x: torch.Tensor,

        normalize=True,

        return_dict=False,

    ):

        if x.ndim != 2:

            raise ValueError(

                f"Expected (B,{self.input_dim})"

            )

        features = self.feature_extractor(

            x.float()

        )

        embedding = self.projection(

            features

        )

        if normalize:

            embedding = F.normalize(

                embedding,

                p=2,

                dim=1,

            )

        if return_dict:

            return {

                "features": features,

                "embedding": embedding,

            }

        return embedding

    # =====================================================
    # Forward
    # =====================================================

    def forward(

        self,

        x,

    ):

        return self.forward_features(x)
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

        params = sum(

            p.numel()

            for p in self.parameters()

        )

        return params * 4 / (1024 ** 2)

    # =====================================================
    # Statistics
    # =====================================================

    def statistics(self):

        total = sum(

            p.numel()

            for p in self.parameters()

        )

        trainable = sum(

            p.numel()

            for p in self.parameters()

            if p.requires_grad

        )

        return {

            "input_dim": self.input_dim,

            "embedding_dim": self.embedding_dim,

            "hidden_dims": self.hidden_dims,

            "dropout": self.dropout_rate,

            "total_parameters": total,

            "trainable_parameters": trainable,

            "size_mb": round(

                self.model_size_mb(),

                2,

            ),

        }

    # =====================================================
    # Print Statistics
    # =====================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("Clinical Encoder Summary")

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
        x: torch.Tensor,
    ):

        self.eval()

        return self.forward(x)

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

        input_shape=(1, 20),

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

            input_names=["clinical_input"],

            output_names=["clinical_embedding"],

            dynamic_axes={

                "clinical_input": {

                    0: "batch"

                },

                "clinical_embedding": {

                    0: "batch"

                },

            },

        )

        print(

            f"ONNX exported to {save_path}"

        )

    # =====================================================
    # Validate Input
    # =====================================================

    def validate_input(
        self,
        x: torch.Tensor,
    ):

        if not isinstance(

            x,

            torch.Tensor,

        ):

            raise TypeError(

                "Input must be torch.Tensor"

            )

        if x.ndim != 2:

            raise ValueError(

                f"Expected shape (B,{self.input_dim})"

            )

        if x.shape[1] != self.input_dim:

            raise ValueError(

                f"Expected {self.input_dim} features, "

                f"received {x.shape[1]}"

            )

        if torch.isnan(x).any():

            raise ValueError(

                "NaN values detected."

            )

        if torch.isinf(x).any():

            raise ValueError(

                "Infinite values detected."

            )

        return True

    # =====================================================
    # Batch Embedding
    # =====================================================

    @torch.no_grad()
    def embed_batch(
        self,
        x: torch.Tensor,
    ):

        self.validate_input(x)

        self.eval()

        embedding = self.forward(x)

        return embedding

    # =====================================================
    # Cosine Similarity
    # =====================================================

    @staticmethod
    def cosine_similarity(

        embedding1: torch.Tensor,

        embedding2: torch.Tensor,

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

            embedding1 * embedding2,

            dim=1,

        )

        return similarity
    # =====================================================
# Factory Function
# =====================================================

def build_clinical_encoder(
    input_dim: int,
    embedding_dim: int = 256,
    hidden_dims=(256, 128),
    dropout: float = 0.30,
):

    """
    Factory function.

    Example
    -------
    encoder = build_clinical_encoder(
        input_dim=20
    )
    """

    return ClinicalEncoder(

        input_dim=input_dim,

        embedding_dim=embedding_dim,

        hidden_dims=hidden_dims,

        dropout=dropout,

    )


# =====================================================
# Dummy Clinical Batch
# =====================================================

def dummy_input(

    batch_size: int = 4,

    input_dim: int = 20,

):

    """
    Generate dummy clinical features.
    """

    return torch.randn(

        batch_size,

        input_dim,

    )


# =====================================================
# Compare Embeddings
# =====================================================

def compare_embeddings(

    embedding1: torch.Tensor,

    embedding2: torch.Tensor,

):

    """
    Euclidean distance.
    """

    return torch.norm(

        embedding1 - embedding2,

        dim=1,

    )


# =====================================================
# Self Test
# =====================================================

def self_test():

    print("=" * 70)

    print("Clinical Encoder Self Test")

    print("=" * 70)

    encoder = build_clinical_encoder(

        input_dim=20,

        embedding_dim=256,

    )

    encoder.print_summary()

    x = dummy_input(

        batch_size=8,

        input_dim=20,

    )

    with torch.no_grad():

        embedding = encoder(x)

    print()

    print("Input Shape")

    print(x.shape)

    print()

    print("Embedding Shape")

    print(embedding.shape)

    assert embedding.shape == (

        8,

        256,

    )

    print()

    print("Self Test Passed")

    print("=" * 70)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    self_test()        
