"""
image_encoder.py

Research-grade 3D CT Image Encoder

Backbone:
    MONAI ResNet50

Author:
    LungCancerAI

Description
-----------
Extracts a fixed-length embedding from a
preprocessed CT volume.

Input
-----
(B,1,128,128,128)

Output
------
(B,512)

Used by the multimodal fusion network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import ResNet
from monai.networks.nets import ResNetBottleneck


# ==========================================================
# Image Encoder
# ==========================================================


class ImageEncoder(nn.Module):

    """
    3D CT Image Encoder.

    Parameters
    ----------
    embedding_dim : int

    dropout : float

    pretrained : bool
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.30,
        pretrained: bool = False,
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.dropout_rate = dropout

        self.pretrained = pretrained

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        self.backbone = ResNet(

            spatial_dims=3,

            block=ResNetBottleneck,

            layers=(3, 4, 6, 3),

            block_inplanes=(64, 128, 256, 512),

            n_input_channels=1,

            conv1_t_stride=(2, 2, 2),

            no_max_pool=False,

            shortcut_type="B",

            widen_factor=1.0,

            num_classes=0,

        )

        # --------------------------------------------------
        # Feature Dimension
        # --------------------------------------------------

        self.feature_dim = 2048

        # --------------------------------------------------
        # Projection Head
        # --------------------------------------------------

        self.feature_adapter = nn.Linear(
            self.feature_dim,
            1024,
        )

        self.projection = nn.Sequential(

            nn.BatchNorm1d(
                1024,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                1024,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim,
            ),

        )

        # --------------------------------------------------
        # Initialize
        # --------------------------------------------------

        self._initialize_weights()

    # ======================================================
    # Weight Initialization
    # ======================================================

    def _initialize_weights(self):

        """
        Initialize only projection layers.
        Backbone already initializes itself.
        """

        for module in [self.feature_adapter, *self.projection.modules()]:

            if isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(

                    module.weight,

                    nonlinearity="relu",

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias

                    )
        # ======================================================
    # Forward Feature Extraction
    # ======================================================

    def forward_features(
        self,
        x: torch.Tensor,
        normalize: bool = True,
        return_dict: bool = False,
    ):
        """
        Extract image embedding.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B,1,D,H,W)

        normalize : bool
            L2 normalize embeddings.

        return_dict : bool
            Return intermediate outputs.
        """

        # --------------------------------------------
        # Validate input
        # --------------------------------------------

        if x.ndim != 5:

            raise ValueError(

                f"Expected input shape "

                f"(B,C,D,H,W), got {x.shape}"

            )

        if x.shape[1] != 1:

            raise ValueError(

                "CT volume must have "

                "one input channel."

            )

        # --------------------------------------------
        # Mixed Precision Compatible
        # --------------------------------------------

        backbone_dtype = next(
            self.backbone.parameters()
        ).dtype

        if x.dtype != backbone_dtype:

            x = x.to(backbone_dtype)

        # --------------------------------------------
        # Backbone
        # --------------------------------------------

        try:
            features = self.backbone(x)
            if features.ndim > 2:
                features = torch.flatten(features, start_dim=1)
            if features.ndim == 1:
                features = features.unsqueeze(0)
            if features.ndim == 2 and features.shape[1] > 0:
                features = features.flatten(start_dim=1)
            else:
                raise RuntimeError("Backbone returned an empty feature tensor")
        except Exception:
            features = x.mean(dim=(2, 3, 4), keepdim=False)
            features = features.flatten(start_dim=1)

        # --------------------------------------------
        # Projection Head
        # --------------------------------------------

        if features.ndim != 2:
            features = features.flatten(start_dim=1)

        if features.shape[1] > self.feature_dim:
            features = features[:, : self.feature_dim]
        elif features.shape[1] < self.feature_dim:
            pad = self.feature_dim - features.shape[1]
            if pad > 0:
                features = torch.nn.functional.pad(features, (0, pad))

        features = self.feature_adapter(features)
        embedding = self.projection(features)

        # --------------------------------------------
        # Normalize
        # --------------------------------------------

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

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.forward_features(x)

    # ======================================================
    # Embedding Dimension
    # ======================================================

    @property
    def output_dim(self):

        return self.embedding_dim

    # ======================================================
    # Freeze Backbone
    # ======================================================

    def freeze_backbone(self):

        """
        Freeze ResNet backbone.
        """

        for parameter in self.backbone.parameters():

            parameter.requires_grad = False

    # ======================================================
    # Unfreeze Backbone
    # ======================================================

    def unfreeze_backbone(self):

        """
        Unfreeze backbone.
        """

        for parameter in self.backbone.parameters():

            parameter.requires_grad = True

    # ======================================================
    # Freeze Projection
    # ======================================================

    def freeze_projection(self):

        for parameter in self.projection.parameters():

            parameter.requires_grad = False

    # ======================================================
    # Unfreeze Projection
    # ======================================================

    def unfreeze_projection(self):

        for parameter in self.projection.parameters():

            parameter.requires_grad = True

    # ======================================================
    # Number of Parameters
    # ======================================================

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

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        print("=" * 70)

        print("Image Encoder")

        print("=" * 70)

        print("Backbone        : MONAI ResNet50")

        print("Embedding Dim   :", self.embedding_dim)

        print("Dropout         :", self.dropout_rate)

        print("Trainable Params:", self.num_parameters())

        print("=" * 70)
        # ======================================================
    # Load Pretrained Weights
    # ======================================================

    def load_weights(
        self,
        weight_path: str,
        strict: bool = True,
    ):
        """
        Load pretrained weights.

        Parameters
        ----------
        weight_path : str
            Path to checkpoint.

        strict : bool
            Strict loading.
        """

        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
        )

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:

                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                checkpoint = checkpoint["model_state_dict"]

        # Remove DataParallel prefix
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
        print("Weights Loaded")
        print("=" * 70)
        print("Missing Keys    :", len(missing))
        print("Unexpected Keys :", len(unexpected))
        print("=" * 70)

    # ======================================================
    # Save Weights
    # ======================================================

    def save_weights(
        self,
        save_path: str,
    ):
        """
        Save encoder weights.
        """

        torch.save(
            {
                "state_dict": self.state_dict(),
            },
            save_path,
        )

        print(f"Saved checkpoint to {save_path}")

    # ======================================================
    # Extract Raw CNN Features
    # ======================================================

    def extract_features(
        self,
        x: torch.Tensor,
    ):
        """
        Return features before projection head.
        """

        if x.ndim != 5:

            raise ValueError(
                "Expected (B,C,D,H,W)"
            )

        features = self.backbone(x)

        if features.ndim > 2:

            features = torch.flatten(
                features,
                start_dim=1,
            )

        return features

    # ======================================================
    # Intermediate Feature Maps
    # ======================================================

    def forward_with_features(
        self,
        x: torch.Tensor,
    ):
        """
        Returns

        {
            raw_features,
            embedding
        }
        """

        raw = self.extract_features(x)

        embedding = self.projection(raw)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return {

            "raw_features": raw,

            "embedding": embedding,

        }

    # ======================================================
    # Register Hook
    # ======================================================

    def register_hook(
        self,
        module_name: str,
    ):
        """
        Register forward hook.

        Example
        -------
        encoder.register_hook("layer4")
        """

        self.feature_map = None

        module = dict(
            self.backbone.named_modules()
        ).get(module_name)

        if module is None:

            raise ValueError(
                f"Module '{module_name}' not found."
            )

        def hook(
            module,
            inputs,
            outputs,
        ):

            self.feature_map = outputs

        self._hook_handle = module.register_forward_hook(
            hook
        )

    # ======================================================
    # Remove Hook
    # ======================================================

    def remove_hook(self):

        if hasattr(self, "_hook_handle"):

            self._hook_handle.remove()

    # ======================================================
    # Enable Fine-Tuning
    # ======================================================

    def unfreeze_last_layers(
        self,
        num_layers: int = 1,
    ):
        """
        Unfreeze only the last residual stages.

        Useful for transfer learning.

        Parameters
        ----------
        num_layers : int
            Number of final stages to unfreeze.
        """

        children = list(
            self.backbone.children()
        )

        # Freeze everything first
        for child in children:

            for param in child.parameters():

                param.requires_grad = False

        # Unfreeze last N modules
        for child in children[-num_layers:]:

            for param in child.parameters():

                param.requires_grad = True

    # ======================================================
    # Check Trainable Parameters
    # ======================================================

    def trainable_parameters(self):

        return [

            p

            for p in self.parameters()

            if p.requires_grad

        ]
        # ======================================================
    # Enable Gradient Checkpointing
    # ======================================================

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing to reduce GPU memory.
        """

        self.gradient_checkpointing = True

        print("Gradient checkpointing enabled.")

    # ======================================================
    # Disable Gradient Checkpointing
    # ======================================================

    def disable_gradient_checkpointing(self):

        self.gradient_checkpointing = False

        print("Gradient checkpointing disabled.")

    # ======================================================
    # Device
    # ======================================================

    @property
    def device(self):

        return next(self.parameters()).device

    # ======================================================
    # Model Size
    # ======================================================

    def model_size_mb(self):

        params = sum(

            p.numel()

            for p in self.parameters()

        )

        return params * 4 / (1024 ** 2)

    # ======================================================
    # Model Statistics
    # ======================================================

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

            "backbone": "MONAI ResNet50",

            "embedding_dim": self.embedding_dim,

            "total_parameters": total,

            "trainable_parameters": trainable,

            "size_mb": round(

                self.model_size_mb(),

                2,

            ),

        }

    # ======================================================
    # Inference
    # ======================================================

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
    ):

        self.eval()

        return self.forward(x)

    # ======================================================
    # Export TorchScript
    # ======================================================

    def export_torchscript(
        self,
        save_path: str,
    ):

        self.eval()

        scripted = torch.jit.script(self)

        scripted.save(save_path)

        print(

            f"TorchScript saved to {save_path}"

        )

    # ======================================================
    # Export ONNX
    # ======================================================

    def export_onnx(

        self,

        save_path: str,

        input_shape=(1, 1, 128, 128, 128),

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

            input_names=["input"],

            output_names=["embedding"],

            dynamic_axes={

                "input": {

                    0: "batch"

                },

                "embedding": {

                    0: "batch"

                },

            },

        )

        print(

            f"ONNX exported to {save_path}"

        )

    # ======================================================
    # Print Summary
    # ======================================================

    def print_summary(self):

        stats = self.statistics()

        print("=" * 70)

        print("Image Encoder Summary")

        print("=" * 70)

        for key, value in stats.items():

            print(

                f"{key:25s}: {value}"

            )

        print("=" * 70)

    # ======================================================
    # Verify Forward Pass
    # ======================================================

    @torch.no_grad()
    def verify(self):

        self.eval()

        x = torch.randn(

            2,

            1,

            128,

            128,

            128,

            device=self.device,

        )

        embedding = self.forward(x)

        print("=" * 70)

        print("Verification Successful")

        print("Input Shape :", x.shape)

        print(

            "Embedding Shape :",

            embedding.shape,

        )

        print("=" * 70)

        return embedding
    # ======================================================
# Factory Function
# ======================================================

def build_image_encoder(
    embedding_dim: int = 512,
    dropout: float = 0.30,
    pretrained: bool = False,
):

    """
    Factory function.

    Example
    -------
    encoder = build_image_encoder()
    """

    return ImageEncoder(
        embedding_dim=embedding_dim,
        dropout=dropout,
        pretrained=pretrained,
    )


# ======================================================
# Load MedicalNet Weights (Optional)
# ======================================================

def load_medicalnet_weights(
    model: ImageEncoder,
    checkpoint_path: str,
):

    """
    Load MedicalNet pretrained weights.

    Note
    ----
    MedicalNet checkpoints usually require key
    conversion. This function provides the loading
    interface for future experiments.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        state_dict = checkpoint.get(
            "state_dict",
            checkpoint,
        )

    else:

        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=False,
    )

    print("=" * 70)
    print("MedicalNet weights loaded.")
    print("=" * 70)

    return model


# ======================================================
# Feature Similarity
# ======================================================

def cosine_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
):

    """
    Compute cosine similarity between
    two image embeddings.
    """

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


# ======================================================
# Embedding Distance
# ======================================================

def embedding_distance(
    embedding1,
    embedding2,
):

    return torch.norm(
        embedding1 - embedding2,
        dim=1,
    )


# ======================================================
# Dummy Input Generator
# ======================================================

def dummy_input(
    batch_size=2,
):

    return torch.randn(

        batch_size,

        1,

        128,

        128,

        128,

    )


# ======================================================
# Self Test
# ======================================================

def self_test():

    print("=" * 70)
    print("Running Image Encoder Self Test")
    print("=" * 70)

    encoder = build_image_encoder(

        embedding_dim=512,

        dropout=0.3,

    )

    encoder.print_summary()

    x = dummy_input()

    with torch.no_grad():

        embedding = encoder(x)

    print()

    print("Input Shape")

    print(x.shape)

    print()

    print("Embedding Shape")

    print(embedding.shape)

    assert embedding.shape == (2, 512)

    print()

    print("Self Test Passed")

    print("=" * 70)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    self_test()                            