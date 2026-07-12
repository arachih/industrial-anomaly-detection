# Industrial anomaly detection with pretrained ResNet features

Two methods for locating defects on manufactured parts, applied to the MVTec AD benchmark. Feature-space anomaly detection using a frozen ResNet-18 backbone — no fine-tuning, no labels beyond `good` samples at training time.

![Pill comparison](results/heatmaps/pill_comparison.png)
*Pill with `color` defect. KNN localizes the specks correctly; PaDiM fires mostly on the object boundary.*

![Grid comparison](results/heatmaps/grid_comparison.png)
*Metal grid with broken wires. PaDiM cleanly points at each defect; KNN barely reacts.*

## What this is

An implementation and a side-by-side comparison of two anomaly detection methods that operate purely in the feature space of a pretrained CNN:

- **PaDiM-style Gaussian model** (Defard et al., 2020) — a per-pixel Mahalanobis distance from a Gaussian fitted on the training features.
- **PatchCore-style memory bank** (Roth et al., 2022) — a position-agnostic nearest-neighbor search in a bank of training feature vectors.

Both use the same intermediate feature map (`layer3` of a frozen ImageNet ResNet-18). Neither is fine-tuned. The point is to see how the *assumptions* each method makes about what "normal" looks like determine which category it handles well.

## The interesting result

The two methods disagree — and which one wins flips between categories.

**Grid** is a homogeneous, position-locked texture. Training grids all look the same at every location, so PaDiM's per-location Gaussians are tight, and any deviation stands out sharply — hence the clean localization of the broken wires. KNN's memory bank contains every training feature vector, but on a repetitive texture a defective patch still finds *some* neighbor in the bank that looks vaguely similar, so the distance stays small and the defect is missed.

**Pill** has natural pose, lighting, and imprint variation between training samples. That spreads the per-location feature distribution: PaDiM's Gaussian at each pixel becomes diffuse and the highest-variance region ends up being the object edge — literally where you don't want it to react. KNN is position-agnostic: a colored speck is compared to training features from *any* position, and finds no close neighbor, so the score is high in the right place.

The one-line rule: PaDiM assumes normal is a single Gaussian blob at each pixel; KNN only assumes normal features live "close to something already seen." When the data matches the assumption, that method wins. This kind of behavior — same backbone, same features, different modelling assumptions producing opposite results — is exactly why running more than one method matters when you actually deploy anomaly detection to a production line.

## Methods, briefly

Both share the same pipeline:

1. Extract features from `layer3` of a pretrained ResNet-18 (256 channels, 14×14 spatial grid at 224×224 input).
2. Fit a model of "normal" using training features from the `good` class only.
3. Score test features per spatial location.
4. Upsample bilinearly to image resolution and smooth with a Gaussian filter.

**`GaussianAD`** (`src/methods.py`)
Fit $\mathcal{N}(\mu_{hw}, \Sigma_{hw})$ per pixel over training features. Score is the Mahalanobis distance from the pixel's own Gaussian. Diagonal covariance by default (per-channel variance); full covariance available via a flag.

**`NearestNeighborAD`** (`src/methods.py`)
Pool all training feature vectors ($N \times H \times W$ of them) into a memory bank, capped at 20 000 randomly-sampled vectors — a cheap approximation of PatchCore's coreset. Score is the mean L2 distance to the $k=3$ nearest neighbors.

## Repository layout

```
src/
dataset.py       MVTec loader, PIL -> tensor, train/test splits
features.py      Frozen ResNet-18 truncated at a configurable layer
methods.py       GaussianAD and NearestNeighborAD, shared interface
visualize.py     Heatmap upsampling, smoothing, side-by-side plotting
notebooks/
01_exploration.ipynb       Look at the data
02_padim_baseline.ipynb    Fit and evaluate GaussianAD
03_knn_method.ipynb        Fit and evaluate NearestNeighborAD
results/heatmaps/            Comparison figures used in this README
```

## Reproducing the figures

You need the MVTec AD dataset — only the `pill` and `grid` categories are used above. Get it from [the MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad).

```bash
git clone git@github.com:arachih/industrial-anomaly-detection.git
cd industrial-anomaly-detection

# Environment (GPU, CUDA 12.4). CPU also works, slower.
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# Point the code at your MVTec data
ln -s /path/to/mvtec data/mvtec

jupyter lab notebooks/
```

On my laptop (RTX 1000 Ada, 6 GB VRAM), fitting either method on one category takes ~30 seconds and scoring a test image is under 100 ms.

## Design choices worth flagging

**Why `layer3` and not `layer4`?** Layer 4 gives more semantic features but only 7×7 spatial resolution — too coarse to localize small defects like pill color specks (a few pixels each). Layer 3 keeps 14×14, which upsamples to a legible heatmap. Layer 2 is finer but the features are too low-level (mostly edges) and lose the ImageNet semantics that make this whole approach work in the first place.

**Why diagonal covariance in `GaussianAD`?** The full 256×256 covariance per pixel means 12M parameters per category and matrix inversion at inference. The diagonal version is ~50× lighter for maybe a percentage point of accuracy on MVTec AD (PaDiM's original paper reports this trade-off).

**Why cap the KNN bank at 20 000?** The natural bank size (267 pill training images × 14 × 14 ≈ 52k vectors) is manageable but slow. PatchCore uses a coreset algorithm (greedy k-center) to pick a representative subsample; here we use random subsampling — coarser but pulls in no extra dependencies and is enough to demonstrate the method.

**Why no ROC/AUROC numbers?** MVTec test sets are ~100 images per category. AUROC on that few samples is noisy, and this repo is about localization quality (visible heatmaps), not classification. A full 15-category sweep with proper AUROC is on the roadmap below.

## Limitations, honestly

- **Not SOTA.** Reference PatchCore uses ResNet-50 with WideResNet weights, coreset subsampling, and multi-scale features (concatenated `layer2` + `layer3`), reaching ~99% AUROC on MVTec AD. This repo gives you most of the intuition with a small fraction of the code — it is meant for learning and comparison, not for shipping to a production line as-is.
- **Only two MVTec categories** are used in the figures. Methods behave differently across the 15 categories; a full sweep is deferred.
- **Assumes clean training data.** Contamination of the `good` training set (subtle defects nobody noticed) degrades both methods significantly.
- **Static images only.** For video streams (e.g. conveyor inspection), the extension is straightforward — temporal smoothing on the score map, persistence rule on consecutive frames — but not implemented here.

## Roadmap

- Full 15-category sweep with per-category AUROC and one qualitative example per defect type.
- Multi-scale features (concatenate `layer2` + `layer3` outputs before scoring).
- Real coreset subsampling in `NearestNeighborAD`.
- Video pipeline: temporal EMA on the score map, N-consecutive-frame trigger, alert via HTTP webhook.
- CFD extension: applying the same "normal-manifold" framework to numerical simulation field snapshots for detecting solver instabilities or extreme events.

## References

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, ICPR 2020.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.

---

*Built as a v1.0 portfolio project after a three-day PyTorch training. The methods here are educational implementations, not production code — see PatchCore's official repository for the state of the art.*
