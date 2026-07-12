"""
Feature extraction from a pretrained ImageNet backbone.

We use ResNet18 by default: small enough to be fast on CPU/small GPU, deep enough
to give features that discriminate texture and shape defects. We tap the network
at ``layer3`` (not the final avg-pool) to keep spatial resolution — anomaly
detection needs a feature map, not a single vector.

Output spatial size depends on input size:
    input 224x224 -> layer3 output  14x14  (256 channels)
    input 256x256 -> layer3 output  16x16  (256 channels)

Reference for the "cut the backbone at layerX" idea: Defard et al., PaDiM,
ICPR 2020; Roth et al., PatchCore, CVPR 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# Which ResNet18 sub-modules count as "layer1..4" for slicing.
# We use named-module extraction rather than list(children())[:N] because it's
# explicit and won't break if torchvision reorders internals.
_LAYER_STOPS = {
    "layer1": ["conv1", "bn1", "relu", "maxpool", "layer1"],
    "layer2": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"],
    "layer3": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"],
    "layer4": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"],
}


class FeatureExtractor(nn.Module):
    """Pretrained ResNet18 truncated at a chosen intermediate layer.

    All parameters are frozen — this module is for inference only.

    Parameters
    ----------
    stop_layer : {"layer1", "layer2", "layer3", "layer4"}, default "layer3"
        Where to cut the backbone. Later layers = more semantic, coarser spatial;
        earlier layers = more texture, finer spatial.
    """

    def __init__(self, stop_layer: str = "layer3"):
        super().__init__()
        if stop_layer not in _LAYER_STOPS:
            raise ValueError(
                f"stop_layer must be one of {list(_LAYER_STOPS)}, got {stop_layer!r}"
            )
        self.stop_layer = stop_layer

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = [getattr(backbone, name) for name in _LAYER_STOPS[stop_layer]]
        self.body = nn.Sequential(*modules)

        for p in self.body.parameters():
            p.requires_grad_(False)

        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) image batch  ->  (B, C, h, w) feature map."""
        return self.body(x)

    @property
    def num_channels(self) -> int:
        """Number of output channels at the chosen stop layer (ResNet18-specific)."""
        return {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}[self.stop_layer]