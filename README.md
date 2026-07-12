# Industrial anomaly detection with pretrained ResNet features

Two methods for locating defects on manufactured parts, applied to the MVTec AD benchmark. Feature-space anomaly detection using a frozen ResNet-18 backbone, no fine-tuning, no labels beyond `good` samples at training time.

![Pill comparison](results/heatmaps/pill_comparison.png)

*Pill with a `color` defect. KNN localizes the specks correctly. PaDiM fires on the object boundary and misses them.*

![Grid comparison](results/heatmaps/grid_comparison.png)

*Grid with broken wires. PaDiM cleanly points at each defect. KNN barely reacts.*

## The interesting result

The two methods disagree, and which one wins flips between categories.

Grid is a repetitive texture with almost no variation between training images. PaDiM's per-pixel Gaussians end up tight, so any deviation stands out. KNN's memory bank always finds *some* neighbor that looks vaguely like the defective patch, so the distance stays small and the defect gets missed.

Pill has natural pose and lighting variation between training samples. That spreads the per-location feature distribution, PaDiM's Gaussian at each pixel becomes diffuse, and the highest-variance region ends up being the object edge, i.e. exactly where you don't want the model to react. KNN is position-agnostic: a colored speck is compared to training features from any location, finds no close match, and lights up in the right place.

The one-line rule: PaDiM assumes normal is a single Gaussian blob at each pixel. KNN only assumes normal features live close to something already seen. When the data matches the assumption, that method wins.

## Methods

Both share the pipeline:

1. Extract features from `layer3` of a pretrained ResNet-18 (256 channels, 14x14 grid at 224x224 input).
2. Fit a model of "normal" using training features from the `good` class only.
3. Score test features per spatial location.
4. Upsample to image resolution and smooth with a Gaussian filter.

`GaussianAD` fits `N(mu, Sigma)` per pixel over training features, scores test images by Mahalanobis distance. Diagonal covariance by default. Follows the shape of Defard et al., PaDiM (ICPR 2020).

`NearestNeighborAD` pools all training feature vectors into a memory bank capped at 20,000 random samples, scores test features by mean L2 distance to the top-3 nearest neighbors. Simplified variant of Roth et al., PatchCore (CVPR 2022), skipping the coreset subsampling.

## Layout

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
## Running it

You need the MVTec AD dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad. The figures above use the `pill` and `grid` categories.

```bash
git clone git@github.com:arachih/industrial-anomaly-detection.git
cd industrial-anomaly-detection

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

ln -s /path/to/mvtec data/mvtec
jupyter lab notebooks/
```

On an RTX 1000 Ada 6 GB, fitting either method on one category takes about 30 seconds and scoring a test image is under 100 ms.

## Design choices

*Why layer3.* Layer 4 gives more semantic features but drops spatial resolution to 7x7, too coarse to localize small defects like the color specks on the pill (a few pixels each). Layer 2 keeps resolution but the features are too low-level, mostly edges.

*Why diagonal covariance.* Full 256x256 covariance per pixel means 12M parameters per category and matrix inversion at inference, for maybe a percentage point of accuracy on MVTec AD.

*Why cap the KNN bank at 20,000.* The natural bank size (267 pill training images x 14 x 14 ~ 52k vectors) is manageable but slow. PatchCore uses a proper coreset algorithm; random subsampling is coarser but enough to demonstrate the method without extra dependencies.

*No AUROC numbers.* MVTec test sets have around 100 images per category. AUROC on that few samples is noisy, and this repo is about localization quality (visible heatmaps), not classification. A proper 15-category sweep is on the roadmap.

## Limitations

Not SOTA. Reference PatchCore uses ResNet-50 with WideResNet weights, proper coreset subsampling, and multi-scale features. This repo gives the intuition with a fraction of the code.

Only two MVTec categories in the figures. Methods behave differently across the 15 categories, a full sweep is deferred.

Assumes clean training data. Contamination of the `good` set with subtle defects degrades both methods.

Static images only. Video extension is straightforward (temporal smoothing on the score map, persistence rule on consecutive frames) but not implemented here.

## Roadmap

- Full 15-category sweep with per-category AUROC.
- Multi-scale features (concatenate `layer2` and `layer3`).
- Real coreset subsampling in `NearestNeighborAD`.
- Video pipeline: temporal EMA, N-frame trigger, HTTP webhook alerts.
- CFD extension: applying the same normal-manifold framework to simulation snapshots for detecting solver instabilities or extreme events.

## References

- Bergmann et al., MVTec AD, CVPR 2019.
- Defard et al., PaDiM, ICPR 2020.
- Roth et al., PatchCore, CVPR 2022.
