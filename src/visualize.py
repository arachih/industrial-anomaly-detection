"""
Visualization utilities for anomaly heatmaps.

The raw anomaly score maps come out at the feature-map resolution (14x14 for
layer3), and we want to overlay them at image resolution. Two-step process:

1. Upsample to the image size with bilinear interpolation.
2. Smooth with a Gaussian filter to remove blockiness.

We also expose a side-by-side plotting helper for the notebook and README.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def upsample_and_smooth(
    score_map: torch.Tensor,
    out_size: int = 224,
    sigma: float = 4.0,
) -> np.ndarray:
    """Upsample a low-res score map to image resolution and smooth it.

    Parameters
    ----------
    score_map : Tensor of shape (H, W), the raw method output.
    out_size : int, final spatial size (assumes square).
    sigma : float, Gaussian smoothing standard deviation in pixels.
    """
    x = score_map.unsqueeze(0).unsqueeze(0).float()                # (1,1,H,W)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    heatmap = x.squeeze().cpu().numpy()
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def tensor_to_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalization and convert (3,H,W) to (H,W,3) in [0,1]."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def plot_anomaly(
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    label: int | None = None,
    method_name: str = "",
    alpha: float = 0.5,
) -> Figure:
    """Side-by-side plot: original image | heatmap overlay.

    Returns the matplotlib Figure so the caller can save or display it.
    """
    img = tensor_to_image(img_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    title = "Input image"
    if label is not None:
        title += f"  (label: {'anomaly' if label == 1 else 'normal'})"
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(heatmap, cmap="jet", alpha=alpha)
    title = "Anomaly heatmap"
    if method_name:
        title += f"  ({method_name})"
    axes[1].set_title(title)
    axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_method_comparison(
    img_tensor: torch.Tensor,
    heatmaps: dict[str, np.ndarray],
    label: int | None = None,
    alpha: float = 0.5,
) -> Figure:
    """One row: original image + one heatmap column per method.

    ``heatmaps`` maps method_name -> (H, W) numpy array.
    """
    img = tensor_to_image(img_tensor)
    n_cols = 1 + len(heatmaps)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img)
    title = "Input"
    if label is not None:
        title += f" ({'anomaly' if label == 1 else 'normal'})"
    axes[0].set_title(title)
    axes[0].axis("off")

    for ax, (name, hmap) in zip(axes[1:], heatmaps.items()):
        ax.imshow(img)
        ax.imshow(hmap, cmap="jet", alpha=alpha)
        ax.set_title(name)
        ax.axis("off")

    fig.tight_layout()
    return fig