"""
MVTec AD dataset loader.

The MVTec Anomaly Detection dataset has the following structure per category:

    <root>/<category>/train/good/*.png         # normal samples only
    <root>/<category>/test/good/*.png          # normal test samples
    <root>/<category>/test/<defect_type>/*.png # anomalous test samples
    <root>/<category>/ground_truth/<defect_type>/*_mask.png  # pixel-level masks

Reference: Bergmann et al., "MVTec AD - A Comprehensive Real-World Dataset
for Unsupervised Anomaly Detection", CVPR 2019.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


# ImageNet normalization constants (matches torchvision pretrained backbones).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """One MVTec category, either train (normal only) or test (mixed).

    Parameters
    ----------
    root_dir : str or Path
        Path to the MVTec root, containing one folder per category.
    category : str
        Category name, e.g. "pill" or "grid".
    is_train : bool, default True
        If True, load only the training normal samples.
        If False, load all test samples (normal + defective) with labels.
    transform : callable, optional
        Torchvision-style transform applied to each PIL image.

    Returns per sample
    ------------------
    (image_tensor, label) where label is 0 for normal, 1 for anomaly.
    In train mode, all labels are 0.
    """

    def __init__(
        self,
        root_dir: str | Path,
        category: str,
        is_train: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.is_train = is_train
        self.transform = transform

        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self._load_index()

        if len(self.image_paths) == 0:
            split = "train/good" if is_train else "test"
            raise FileNotFoundError(
                f"No images found for '{category}' in {self.root_dir / category / split}. "
                f"Check the MVTec directory layout."
            )

    def _load_index(self) -> None:
        cat_dir = self.root_dir / self.category
        if self.is_train:
            paths = sorted(glob.glob(str(cat_dir / "train" / "good" / "*.png")))
            self.image_paths = paths
            self.labels = [0] * len(paths)
        else:
            paths = sorted(glob.glob(str(cat_dir / "test" / "*" / "*.png")))
            self.image_paths = paths
            self.labels = [0 if "/good/" in p else 1 for p in paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def defect_types(self) -> list[str]:
        """Return the unique defect subfolder names (test mode only)."""
        if self.is_train:
            return ["good"]
        return sorted({Path(p).parent.name for p in self.image_paths})