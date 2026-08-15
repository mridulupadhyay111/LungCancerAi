"""
resnet3d.py

Research-grade 3D ResNet backbone for LungCancerAI.

Supports
--------
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

Designed for volumetric CT scans.

Author: LungCancerAI
"""

from __future__ import annotations

from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helper Functions
# ============================================================

def conv3x3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv3d:
    """
    3×3×3 convolution.
    """

    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def conv1x1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
) -> nn.Conv3d:
    """
    1×1×1 convolution.
    """

    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


# ============================================================
# Residual Basic Block
# ============================================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        if groups != 1:
            raise ValueError(
                "BasicBlock supports groups=1 only."
            )

        if base_width != 64:
            raise ValueError(
                "BasicBlock supports base_width=64 only."
            )

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported."
            )

        self.conv1 = conv3x3x3(
            inplanes,
            planes,
            stride,
        )

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(
            planes,
            planes,
        )

        self.bn2 = norm_layer(planes)

        self.downsample = downsample

        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


# ============================================================
# Bottleneck Block
# ============================================================

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        width = int(
            planes * (base_width / 64.0)
        ) * groups

        self.conv1 = conv1x1x1(
            inplanes,
            width,
        )

        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3x3(
            width,
            width,
            stride,
            groups,
            dilation,
        )

        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1x1(
            width,
            planes * self.expansion,
        )

        self.bn3 = norm_layer(
            planes * self.expansion
        )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out
        # ============================================================
# ResNet3D
# ============================================================

class ResNet3D(nn.Module):
    """
    Research-grade 3D ResNet.

    Parameters
    ----------
    block:
        BasicBlock or Bottleneck

    layers:
        Number of blocks in each stage.

    in_channels:
        Number of input channels.

    num_classes:
        Optional classifier output.

    include_top:
        Whether to include classification head.

    zero_init_residual:
        Zero initialize residual branch.

    feature_dim:
        Output embedding dimension.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 1000,
        include_top: bool = False,
        feature_dim: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self._norm_layer = norm_layer

        self.include_top = include_top

        self.feature_dim = feature_dim

        self.inplanes = 64

        self.dilation = 1

        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [
                False,
                False,
                False,
            ]

        if len(replace_stride_with_dilation) != 3:

            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple."
            )

        self.groups = groups

        self.base_width = width_per_group

        # -----------------------------------------------------
        # Stem
        # -----------------------------------------------------

        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = norm_layer(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # -----------------------------------------------------
        # Residual Stages
        # -----------------------------------------------------

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
        )

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        # -----------------------------------------------------
        # Pooling
        # -----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.embedding = nn.Linear(
            512 * block.expansion,
            feature_dim,
        )

        if include_top:

            self.fc = nn.Linear(
                feature_dim,
                num_classes,
            )

        # -----------------------------------------------------
        # Weight Initialization
        # -----------------------------------------------------

        for m in self.modules():

            if isinstance(m, nn.Conv3d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                m,
                (nn.BatchNorm3d, nn.GroupNorm),
            ):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

        # -----------------------------------------------------
        # Zero-init residual
        # -----------------------------------------------------

        if zero_init_residual:

            for m in self.modules():

                if isinstance(
                    m,
                    Bottleneck,
                ):

                    nn.init.constant_(
                        m.bn3.weight,
                        0,
                    )

                elif isinstance(
                    m,
                    BasicBlock,
                ):

                    nn.init.constant_(
                        m.bn2.weight,
                        0,
                    )

    # ============================================================
    # Build One Residual Stage
    # ============================================================

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ):

        norm_layer = self._norm_layer

        downsample = None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if (
            stride != 1
            or self.inplanes != planes * block.expansion
        ):

            downsample = nn.Sequential(

                conv1x1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                ),

                norm_layer(
                    planes * block.expansion
                ),

            )

        layers = []

        layers.append(

            block(

                self.inplanes,

                planes,

                stride,

                downsample,

                self.groups,

                self.base_width,

                previous_dilation,

                norm_layer,

            )

        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(

                block(

                    self.inplanes,

                    planes,

                    groups=self.groups,

                    base_width=self.base_width,

                    dilation=self.dilation,

                    norm_layer=norm_layer,

                )

            )

        return nn.Sequential(*layers)
            # ============================================================
    # Forward Feature Extraction
    # ============================================================

    def forward_features(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """
        Extract deep image features.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, D, H, W)

        return_intermediate : bool
            If True, returns feature maps from each stage.
        """

        # -----------------------------
        # Stem
        # -----------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        stem = x

        x = self.maxpool(x)

        # -----------------------------
        # Residual stages
        # -----------------------------

        c1 = self.layer1(x)

        c2 = self.layer2(c1)

        c3 = self.layer3(c2)

        c4 = self.layer4(c3)

        # -----------------------------
        # Global Average Pooling
        # -----------------------------

        pooled = self.avgpool(c4)

        pooled = torch.flatten(
            pooled,
            start_dim=1,
        )

        embedding = self.embedding(pooled)

        if return_intermediate:

            return {

                "stem": stem,

                "layer1": c1,

                "layer2": c2,

                "layer3": c3,

                "layer4": c4,

                "embedding": embedding

            }

        return embedding

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        embedding = self.forward_features(x)

        if self.include_top:

            logits = self.fc(embedding)

            return logits

        return embedding

    # ============================================================
    # Freeze Backbone
    # ============================================================

    def freeze_backbone(self):

        """
        Freeze all backbone parameters.
        """

        for parameter in self.parameters():

            parameter.requires_grad = False

    # ============================================================
    # Unfreeze Backbone
    # ============================================================

    def unfreeze_backbone(self):

        """
        Unfreeze all parameters.
        """

        for parameter in self.parameters():

            parameter.requires_grad = True

    # ============================================================
    # Freeze Until Layer
    # ============================================================

    def freeze_until(
        self,
        layer_name: str,
    ):

        """
        Example
        -------

        model.freeze_until("layer3")
        """

        freeze = True

        for name, module in self.named_children():

            if name == layer_name:

                freeze = False

            for parameter in module.parameters():

                parameter.requires_grad = not freeze

    # ============================================================
    # Count Parameters
    # ============================================================

    def num_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:

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

    # ============================================================
    # Summary
    # ============================================================

    def summary(self):

        print("=" * 70)

        print("ResNet3D Backbone")

        print("=" * 70)

        print("Feature Dimension :", self.feature_dim)

        print("Include Top       :", self.include_top)

        print("Trainable Params  :", self.num_parameters())

        print("Total Params      :", self.num_parameters(False))

        print("=" * 70)
        # ============================================================
# Factory Functions
# ============================================================

def _resnet(
    block,
    layers,
    **kwargs,
):
    """
    Internal ResNet factory.
    """

    model = ResNet3D(
        block=block,
        layers=layers,
        **kwargs,
    )

    return model


# ============================================================
# ResNet18
# ============================================================

def resnet18_3d(
    **kwargs,
):

    return _resnet(
        BasicBlock,
        [2, 2, 2, 2],
        **kwargs,
    )


# ============================================================
# ResNet34
# ============================================================

def resnet34_3d(
    **kwargs,
):

    return _resnet(
        BasicBlock,
        [3, 4, 6, 3],
        **kwargs,
    )


# ============================================================
# ResNet50
# ============================================================

def resnet50_3d(
    **kwargs,
):

    return _resnet(
        Bottleneck,
        [3, 4, 6, 3],
        **kwargs,
    )


# ============================================================
# ResNet101
# ============================================================

def resnet101_3d(
    **kwargs,
):

    return _resnet(
        Bottleneck,
        [3, 4, 23, 3],
        **kwargs,
    )


# ============================================================
# ResNet152
# ============================================================

def resnet152_3d(
    **kwargs,
):

    return _resnet(
        Bottleneck,
        [3, 8, 36, 3],
        **kwargs,
    )


# ============================================================
# Checkpoint Utilities
# ============================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
):
    """
    Load model checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model instance.

    checkpoint_path : str
        Path to checkpoint.

    strict : bool
        Whether to enforce exact key matching.
    """

    checkpoint = torch.load(
        checkpoint_path,
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

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=strict,
    )

    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)

    print("Missing Keys :", len(missing))
    print("Unexpected Keys :", len(unexpected))

    return model


# ============================================================
# Save Checkpoint
# ============================================================

def save_checkpoint(
    model: nn.Module,
    path: str,
):

    torch.save(
        {
            "state_dict": model.state_dict()
        },
        path,
    )

    print(f"Checkpoint saved to {path}")


# ============================================================
# Feature Extractor Wrapper
# ============================================================

class ResNet3DFeatureExtractor(nn.Module):
    """
    Thin wrapper around ResNet3D for feature extraction.
    Useful when the multimodal model expects only image embeddings.
    """

    def __init__(
        self,
        backbone: ResNet3D,
    ):
        super().__init__()

        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.backbone.forward_features(x)
        # ============================================================
# Feature Hook Manager
# ============================================================

class FeatureHook:
    """
    Register forward hooks to capture intermediate
    feature maps during inference.

    Example
    -------
    >>> model = resnet50_3d()
    >>> hook = FeatureHook(model.layer4)
    >>> output = model(x)
    >>> features = hook.features
    """

    def __init__(self, module: nn.Module):

        self.features = None

        self.handle = module.register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(
        self,
        module,
        inputs,
        outputs,
    ):

        self.features = outputs.detach()

    def close(self):

        self.handle.remove()


# ============================================================
# Register Multiple Hooks
# ============================================================

class FeatureExtractor:
    """
    Extract feature maps from multiple layers.
    """

    def __init__(
        self,
        model: ResNet3D,
        layers=None,
    ):

        self.model = model

        self.outputs = {}

        if layers is None:

            layers = [
                "layer1",
                "layer2",
                "layer3",
                "layer4",
            ]

        self.handles = []

        for layer in layers:

            module = getattr(model, layer)

            handle = module.register_forward_hook(
                self._save_output(layer)
            )

            self.handles.append(handle)

    def _save_output(self, name):

        def fn(module, inp, out):

            self.outputs[name] = out.detach()

        return fn

    def remove(self):

        for h in self.handles:

            h.remove()


# ============================================================
# Enable Gradient Checkpointing
# ============================================================

def enable_gradient_checkpointing(
    model: ResNet3D,
):
    """
    Reduce GPU memory usage by enabling checkpointing.
    """

    from torch.utils.checkpoint import checkpoint_sequential

    def custom_forward(x):

        segments = 4

        modules = [

            model.layer1,

            model.layer2,

            model.layer3,

            model.layer4,

        ]

        return checkpoint_sequential(
            modules,
            segments,
            x,
        )

    model.gradient_checkpointing = True

    model.checkpoint_forward = custom_forward

    return model


# ============================================================
# Disable Gradient Checkpointing
# ============================================================

def disable_gradient_checkpointing(
    model: ResNet3D,
):

    model.gradient_checkpointing = False

    if hasattr(model, "checkpoint_forward"):

        del model.checkpoint_forward

    return model


# ============================================================
# Input Validator
# ============================================================

def validate_input(
    x: torch.Tensor,
):

    if not isinstance(
        x,
        torch.Tensor,
    ):

        raise TypeError(
            "Input must be torch.Tensor"
        )

    if x.ndim != 5:

        raise ValueError(
            "Expected shape (B,C,D,H,W)"
        )

    if torch.isnan(x).any():

        raise ValueError(
            "NaN detected in input."
        )

    if torch.isinf(x).any():

        raise ValueError(
            "Inf detected in input."
        )

    return True


# ============================================================
# Model Statistics
# ============================================================

def count_parameters(
    model: nn.Module,
):

    return sum(

        p.numel()

        for p in model.parameters()

    )


def count_trainable_parameters(
    model: nn.Module,
):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ============================================================
# Print Model Statistics
# ============================================================

def print_model_statistics(
    model: nn.Module,
):

    total = count_parameters(model)

    trainable = count_trainable_parameters(model)

    print("=" * 70)

    print("Model Statistics")

    print("=" * 70)

    print(f"Total Parameters      : {total:,}")

    print(f"Trainable Parameters  : {trainable:,}")

    print(f"Frozen Parameters     : {total-trainable:,}")

    print("=" * 70)


# ============================================================
# Estimate Embedding Dimension
# ============================================================

def get_embedding_dimension(
    model: ResNet3D,
):

    return model.feature_dim


# ============================================================
# Dummy Forward Test
# ============================================================

def sanity_check(
    model: ResNet3D,
    device="cpu",
):

    model.eval()

    model.to(device)

    x = torch.randn(
        2,
        1,
        128,
        128,
        128,
        device=device,
    )

    with torch.no_grad():

        y = model(x)

    print("=" * 70)

    print("Sanity Check Passed")

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)

    print("=" * 70)

    return y
    # ============================================================
# ONNX Export
# ============================================================

def export_onnx(
    model: nn.Module,
    save_path: str,
    input_shape=(1, 1, 128, 128, 128),
):

    model.eval()

    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )

    print(f"ONNX model exported to {save_path}")


# ============================================================
# TorchScript Export
# ============================================================

def export_torchscript(
    model: nn.Module,
    save_path: str,
):

    model.eval()

    scripted = torch.jit.script(model)

    scripted.save(save_path)

    print(f"TorchScript exported to {save_path}")


# ============================================================
# FLOPs Estimation
# ============================================================

def estimate_model_size(
    model: nn.Module,
):

    parameters = count_parameters(model)

    trainable = count_trainable_parameters(model)

    size_mb = parameters * 4 / (1024 ** 2)

    return {

        "parameters": parameters,

        "trainable": trainable,

        "size_mb": round(size_mb, 2)

    }


# ============================================================
# Backbone Registry
# ============================================================

BACKBONES = {

    "resnet18": resnet18_3d,

    "resnet34": resnet34_3d,

    "resnet50": resnet50_3d,

    "resnet101": resnet101_3d,

    "resnet152": resnet152_3d,

}


# ============================================================
# Factory
# ============================================================

def build_backbone(
    name: str = "resnet50",
    **kwargs,
):

    name = name.lower()

    if name not in BACKBONES:

        raise ValueError(

            f"Unknown backbone: {name}"

        )

    return BACKBONES[name](**kwargs)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LungCancerAI")
    print("Research-grade 3D ResNet Backbone")
    print("=" * 70)

    model = build_backbone(

        "resnet50",

        in_channels=1,

        feature_dim=512,

        include_top=False,

    )

    print_model_statistics(model)

    x = torch.randn(

        2,

        1,

        128,

        128,

        128,

    )

    with torch.no_grad():

        embedding = model(x)

    print()

    print("Input Shape")

    print(x.shape)

    print()

    print("Embedding Shape")

    print(embedding.shape)

    print()

    print(

        estimate_model_size(model)

    )

    print("=" * 70)