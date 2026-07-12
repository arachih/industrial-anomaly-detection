"""
Feature-space anomaly detection methods.

Each method builds a "normal distribution model" from training features
(images of the ``good`` class only), then scores test images by their distance
from that model. The score is per spatial location, so we get a heatmap
localizing *where* the anomaly is, not just a global yes/no.

Two methods are implemented:

1. GaussianAD (PaDiM-style)
   Model each spatial location (h, w) as an independent multivariate Gaussian
   over the C feature channels. Score = Mahalanobis distance. Compact model
   (one mean vector and one covariance per pixel). Fast.
   Reference: Defard et al., PaDiM, ICPR 2020.

2. NearestNeighborAD (PatchCore-style, simplified)
   Keep a memory bank of all training feature vectors. Score a test patch by
   its Euclidean distance to the k nearest neighbors in the bank.
   No distributional assumption — works better on multi-modal normal data.
   Reference: Roth et al., PatchCore, CVPR 2022 (this is a stripped-down variant
   without the coreset subsampling).

Both classes share the same interface:
    method.fit(train_loader)                   -> builds the model
    scores = method.score(test_feature_map)    -> per-pixel anomaly scores
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AnomalyMethod(ABC):
    """Common interface for feature-space anomaly detectors."""

    @abstractmethod
    def fit(self, feature_extractor, loader: DataLoader, device: torch.device) -> None:
        """Build the normal model from ``good`` training samples."""
        ...

    @abstractmethod
    def score(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Score a single feature map.

        Parameters
        ----------
        feature_map : Tensor of shape (C, H, W)

        Returns
        -------
        Tensor of shape (H, W) with per-location anomaly scores (higher = more anomalous).
        """
        ...


# ---------------------------------------------------------------------------
# Method 1 — Per-pixel Gaussian (PaDiM-style)
# ---------------------------------------------------------------------------

class GaussianAD(AnomalyMethod):
    """Per-pixel multivariate Gaussian model.

    For each spatial location (h, w) independently, we fit N(mu_hw, Sigma_hw)
    over the C-dimensional feature vectors of the training images. At inference,
    the anomaly score for a location is the Mahalanobis distance from that
    location's fitted distribution.

    Parameters
    ----------
    eps : float
        Ridge added to the diagonal of the covariance for numerical stability.
    diag_only : bool, default True
        If True, use only diagonal covariance (much faster, minor accuracy cost).
        If False, use the full covariance matrix per pixel (slower, more memory).
    """

    def __init__(self, eps: float = 1e-3, diag_only: bool = True):
        self.eps = eps
        self.diag_only = diag_only
        self.mean: torch.Tensor | None = None       # (C, H, W)
        self.inv_cov: torch.Tensor | None = None    # diag: (C, H, W)  or full: (H*W, C, C)

    def fit(self, feature_extractor, loader: DataLoader, device: torch.device) -> None:
        feats = []
        for images, _ in loader:
            f = feature_extractor(images.to(device)).cpu()
            feats.append(f)
        feats = torch.cat(feats, dim=0)                 # (N, C, H, W)
        N, C, H, W = feats.shape

        self.mean = feats.mean(dim=0)                   # (C, H, W)

        if self.diag_only:
            var = feats.var(dim=0, unbiased=False)      # (C, H, W)
            self.inv_cov = 1.0 / (var + self.eps)       # (C, H, W)
        else:
            # Full covariance per spatial location: (H*W, N, C) -> (H*W, C, C)
            f = feats.permute(2, 3, 0, 1).reshape(H * W, N, C)
            centered = f - f.mean(dim=1, keepdim=True)
            cov = centered.transpose(1, 2) @ centered / (N - 1)   # (H*W, C, C)
            cov += self.eps * torch.eye(C).unsqueeze(0)
            self.inv_cov = torch.linalg.inv(cov)

    def score(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("GaussianAD.score called before .fit()")
        fm = feature_map.cpu()                          # (C, H, W)
        delta = fm - self.mean                          # (C, H, W)

        if self.diag_only:
            d2 = (delta * delta * self.inv_cov).sum(dim=0)    # (H, W)
        else:
            C, H, W = delta.shape
            d = delta.permute(1, 2, 0).reshape(H * W, C, 1)   # (H*W, C, 1)
            d2 = (d.transpose(1, 2) @ self.inv_cov @ d).view(H, W)

        return d2.sqrt()


# ---------------------------------------------------------------------------
# Method 2 — Nearest-neighbor in feature space (PatchCore-style)
# ---------------------------------------------------------------------------

class NearestNeighborAD(AnomalyMethod):
    """Memory-bank nearest-neighbor scoring.

    Every training feature vector (C-dim, from any (image, location)) is stored
    in a bank. A test vector's score is its mean distance to the k nearest
    entries in the bank. Position-agnostic: a test location can be compared to
    training vectors from *any* location.

    This makes no Gaussian assumption, which helps when the normal data is
    multi-modal (e.g., pills with different colored specks).

    Parameters
    ----------
    k : int
        Number of nearest neighbors to average.
    max_bank_size : int, optional
        If set, randomly subsample the bank to this many vectors after fitting
        (rough approximation of PatchCore's coreset).
    """

    def __init__(self, k: int = 3, max_bank_size: int | None = 20_000):
        self.k = k
        self.max_bank_size = max_bank_size
        self.bank: torch.Tensor | None = None      # (M, C)

    def fit(self, feature_extractor, loader: DataLoader, device: torch.device) -> None:
        vecs = []
        for images, _ in loader:
            f = feature_extractor(images.to(device)).cpu()    # (B, C, H, W)
            B, C, H, W = f.shape
            v = f.permute(0, 2, 3, 1).reshape(B * H * W, C)   # (B*H*W, C)
            vecs.append(v)
        bank = torch.cat(vecs, dim=0)                          # (N*H*W, C)

        if self.max_bank_size is not None and bank.shape[0] > self.max_bank_size:
            idx = torch.randperm(bank.shape[0])[: self.max_bank_size]
            bank = bank[idx]

        self.bank = bank                                       # (M, C)

    def score(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("NearestNeighborAD.score called before .fit()")
        C, H, W = feature_map.shape
        test = feature_map.permute(1, 2, 0).reshape(H * W, C).cpu()   # (H*W, C)

        # Chunked pairwise distances to avoid a big (H*W, M) matrix on GPU.
        dists = torch.cdist(test, self.bank)                          # (H*W, M)
        topk, _ = dists.topk(self.k, dim=1, largest=False)            # (H*W, k)
        score = topk.mean(dim=1).view(H, W)                           # (H, W)
        return score