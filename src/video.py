"""
Video pipeline for feature-space anomaly detection.

Read an input mp4, score each frame with a pre-fit anomaly method, overlay the
per-pixel heatmap on the original frame, apply a persistence rule to gate an
"ANOMALY" banner, and write an annotated mp4.

Input resolution to the network is 224x224 square (matches the notebook and
training). Display resolution matches the original video, so the heatmap chain
is: 14x14 feature-map score -> 224x224 (bilinear + Gaussian smooth) ->
original size (bilinear). The overlay is blended on the raw frame, not on the
squished one, so the video looks native.

Threshold and method are loaded from a state dict saved by
notebooks/04_corn_custom.ipynb (see results/corn_model.pt).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.features import FeatureExtractor
from src.methods import AnomalyMethod, GaussianAD, NearestNeighborAD


# ---------------------------------------------------------------------------
# Frame preprocessing (matches the notebook's transform pipeline)
# ---------------------------------------------------------------------------

def build_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Per-frame scoring
# ---------------------------------------------------------------------------

def score_frame(
    frame_bgr: np.ndarray,
    extractor: FeatureExtractor,
    method: AnomalyMethod,
    tfm,
    device: torch.device,
    smooth_sigma: float = 4.0,
    input_size: int = 224,
) -> np.ndarray:
    """Score one BGR frame and return a heatmap at input_size x input_size."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = tfm(frame_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        fmap = extractor(img_tensor)[0]
    score = method.score(fmap)   # (H, W) at feature-map resolution

    # Upsample to input_size, smooth
    x = score.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    heatmap = x.squeeze().cpu().numpy()
    if smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)
    return heatmap


# ---------------------------------------------------------------------------
# Overlay + banner
# ---------------------------------------------------------------------------

def apply_overlay(
    frame_bgr: np.ndarray,
    heatmap_small: np.ndarray,
    threshold: float,
    alpha: float = 0.4,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Resize heatmap to the frame size, apply a colormap on a fixed scale, blend.

    vmin/vmax define the color scale (score domain). If not given, we default
    to a scale where the threshold maps roughly to the yellow-green transition:
        vmin = 0
        vmax = 2 * threshold
    So scores below threshold are cool (blue/green), around threshold are
    yellow, well above threshold are red. Consistent across frames.
    """
    h, w = frame_bgr.shape[:2]
    heat = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_LINEAR)

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 2.0 * threshold

    heat_clipped = np.clip((heat - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    heat_uint8 = (heat_clipped * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)

    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_color, alpha, 0.0)
    return blended

def draw_banner(frame: np.ndarray, text: str = "ANOMALY") -> np.ndarray:
    """Draw a red banner with white text at the top of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = max(40, h // 20)
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 200), thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = banner_h / 40.0
    thickness = max(2, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    org = ((w - tw) // 2, (banner_h + th) // 2)
    cv2.putText(out, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def annotate_video(
    input_path: str | Path,
    output_path: str | Path,
    model_state_path: str | Path,
    persistence_frames: int = 10,
    alpha: float = 0.4,
    smooth_sigma: float = 4.0,
    max_frames: int | None = None,
    verbose: bool = True,
):
    """Read input_path, annotate every frame, write to output_path.

    Parameters
    ----------
    persistence_frames : int
        Number of consecutive frames whose max-score must exceed the threshold
        before the ANOMALY banner appears (and disappears again).
    alpha : float in [0, 1]
        Heatmap overlay opacity.
    max_frames : int, optional
        Cap on number of frames processed. Useful for quick tests.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device: {device}")

    # Load fit method + threshold. State can hold either a Gaussian or a KNN model.
    state = torch.load(model_state_path, map_location="cpu", weights_only=False)
    threshold = float(state["threshold"])
    image_size = int(state["image_size"])
    method_name = state.get("method", "knn")

    if method_name == "gaussian":
        method = GaussianAD(diag_only=state["gad_diag_only"])
        method.mean = state["gad_mean"]
        method.inv_cov = state["gad_inv_cov"]
        model_info = f"mean shape: {tuple(method.mean.shape)}"
    elif method_name == "knn":
        method = NearestNeighborAD(k=state["knn_k"])
        method.bank = state["knn_bank"]
        model_info = f"bank: {tuple(method.bank.shape)}"
    else:
        raise ValueError(f"Unknown method in state: {method_name!r}")

    if verbose:
        print(f"method: {method_name}, threshold: {threshold:.3f}, input size: {image_size}, {model_info}")
    
    extractor = FeatureExtractor("layer3").to(device)
    tfm = build_transform(image_size)

    # Video I/O
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"input: {w}x{h} @ {fps:.2f}fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Persistence state: rolling window of "was above threshold" booleans
    hits = deque(maxlen=persistence_frames)
    min_hits = max(1, int(persistence_frames * 0.7))  # 7 out of 10 by default

    n_written = 0
    n_anomaly_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        heatmap_small = score_frame(frame, extractor, method, tfm, device, smooth_sigma, image_size)
        max_score = float(heatmap_small.max())

        hits.append(max_score > threshold)
        show_banner = len(hits) == persistence_frames and sum(hits) >= min_hits

        out_frame = apply_overlay(frame, heatmap_small, threshold=threshold, alpha=alpha)
            
        if show_banner:
            out_frame = draw_banner(out_frame, "ANOMALY")
            n_anomaly_frames += 1

        writer.write(out_frame)
        n_written += 1

        if verbose and n_written % 30 == 0:
            print(f"  frame {n_written}/{total}  max_score={max_score:.2f}  "
                  f"hits={sum(hits)}/{len(hits)}  banner={'ON' if show_banner else 'off'}")

        if max_frames is not None and n_written >= max_frames:
            break

    cap.release()
    writer.release()
    if verbose:
        print(f"done: wrote {n_written} frames, banner on {n_anomaly_frames} of them, output: {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Annotate a video with anomaly heatmaps.")
    p.add_argument("--input", required=True, help="Input mp4 path")
    p.add_argument("--output", required=True, help="Output mp4 path")
    p.add_argument("--model", required=True, help="Model state .pt path")
    p.add_argument("--persistence", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--smooth-sigma", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    annotate_video(
        input_path=args.input,
        output_path=args.output,
        model_state_path=args.model,
        persistence_frames=args.persistence,
        alpha=args.alpha,
        smooth_sigma=args.smooth_sigma,
        max_frames=args.max_frames,
    )