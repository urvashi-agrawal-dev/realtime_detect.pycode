"""
Real-time / CLI deepfake detection for images and videos.

- CLI:  python realtime_detect.py --path "file.mp4"
- Interactive: python realtime_detect.py  (then enter paths; 'q' to quit)

Video aggregation: drop ambiguous (low-confidence) frames, then blend median + weighted
mean of P(fake); compare to calibrated threshold_fake from meta.json.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import models, transforms

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class FaceDetectionTransform:
    def __init__(self, mtcnn: MTCNN, img_size: int = 224):
        self.mtcnn = mtcnn
        self.img_size = img_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Detect face
        try:
            boxes, _ = self.mtcnn.detect(img)
        except Exception:
            boxes = None
        if boxes is not None and len(boxes) > 0:
            # Use the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = box
            # Add some margin
            margin = 0.1
            w = x2 - x1
            h = y2 - y1
            x1 = max(0, x1 - margin * w)
            y1 = max(0, y1 - margin * h)
            x2 = min(img.width, x2 + margin * w)
            y2 = min(img.height, y2 + margin * h)
            face = img.crop((x1, y1, x2, y2))
        else:
            # No face detected, use full image
            face = img
        # Resize to img_size
        face = face.resize((self.img_size, self.img_size), Image.BILINEAR)
        return face


def build_model(arch: str) -> nn.Module:
    arch = arch.lower()
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    raise ValueError("arch must be efficientnet_b0 or resnet50")


def resolve_file(p: str) -> Path:
    return Path(p).expanduser().resolve()


def sample_frame_indices(total_frames: int, n: int) -> List[int]:
    if total_frames <= 0:
        return []
    n = max(1, min(n, total_frames))
    idx = np.linspace(0, total_frames - 1, n, dtype=int)
    return np.unique(idx).tolist()


@dataclass
class VideoDebugResult:
    prediction_label: str
    prob_score: float
    threshold_fake: float
    n_sampled: int
    n_used_after_filter: int
    prob_min: float
    prob_max: float
    prob_mean: float
    prob_std: float
    prob_median: float
    prob_weighted_mean: float
    majority_fake_fraction: float
    mean_prob: float
    max_prob: float
    fake_frame_ratio: float
    decision_reason: str


def _aggregate_frame_probs(
    probs: np.ndarray,
    confidence_margin: float,
    w_median: float,
    w_weighted: float,
) -> Tuple[float, np.ndarray]:
    """
    Drop ambiguous frames (|p-0.5| < margin). If too few remain, use all frames.
    Score = normalized blend of median(kept) and weighted-mean(kept), weights = |p-0.5|.
    """
    if probs.size == 0:
        return 0.5, probs

    margin = max(0.0, float(confidence_margin))
    conf = np.abs(probs - 0.5)
    mask = conf >= margin
    kept = probs[mask]

    min_kept = max(3, int(np.ceil(0.25 * probs.size)))
    if kept.size < min_kept:
        kept = probs.copy()

    w = np.abs(kept - 0.5)
    w = np.maximum(w, 1e-6)

    med = float(np.median(kept))
    wmean = float(np.average(kept, weights=w))
    wm = float(w_median + w_weighted)
    if wm <= 0:
        wm = 1.0
    score = (w_median * med + w_weighted * wmean) / wm
    return float(score), kept


@torch.no_grad()
def predict_image(
    model: nn.Module,
    tf: transforms.Compose,
    device: torch.device,
    img: Image.Image,
    threshold_fake: float,
) -> Tuple[int, float]:
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    prob_fake = float(probs[0])
    pred = 0 if prob_fake >= threshold_fake else 1
    return pred, prob_fake


@torch.no_grad()
def predict_video(
    model: nn.Module,
    tf: transforms.Compose,
    device: torch.device,
    video_path: Path,
    n_frames: int,
    threshold_fake: float,
    confidence_margin: float,
    w_median: float,
    w_weighted: float,
) -> VideoDebugResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, n_frames)

    probs_list: List[float] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tf(img).unsqueeze(0).to(device)
        logits = model(x)
        p_fake = torch.softmax(logits, dim=1)[0, 0].item()
        probs_list.append(float(p_fake))

    cap.release()
    arr = np.array(probs_list, dtype=np.float64)
    n_sampled = int(arr.size)

    if n_sampled == 0:
        return VideoDebugResult(
            prediction_label="REAL",
            prob_score=0.0,
            threshold_fake=threshold_fake,
            n_sampled=0,
            n_used_after_filter=0,
            prob_min=0.0,
            prob_max=0.0,
            prob_mean=0.0,
            prob_std=0.0,
            prob_median=0.0,
            prob_weighted_mean=0.0,
            majority_fake_fraction=0.0,
            mean_prob=0.0,
            max_prob=0.0,
            fake_frame_ratio=0.0,
            decision_reason="No frames sampled",
        )

    vmin = float(arr.min())
    vmax = float(arr.max())
    mean = float(arr.mean())
    std = float(arr.std())
    med_all = float(np.median(arr))

    # New metrics
    mean_prob = mean
    max_prob = vmax
    fake_frame_ratio = float((arr >= 0.5).sum() / arr.size)

    # New decision logic
    is_fake = (
        mean_prob > 0.25
        or max_prob > 0.6
        or fake_frame_ratio > 0.2
    )
    pred = 0 if is_fake else 1
    label = "FAKE" if pred == 0 else "REAL"

    # Determine reason
    reasons = []
    if mean_prob > 0.25:
        reasons.append(f"mean_prob={mean_prob:.3f}>0.25")
    if max_prob > 0.6:
        reasons.append(f"max_prob={max_prob:.3f}>0.6")
    if fake_frame_ratio > 0.2:
        reasons.append(f"fake_frame_ratio={fake_frame_ratio:.3f}>0.2")
    if not reasons:
        reasons.append("none of the conditions met")
    decision_reason = "; ".join(reasons)

    # Old aggregation for compatibility
    score, kept = _aggregate_frame_probs(arr, confidence_margin, w_median, w_weighted)
    n_used = int(kept.size)
    w_arr = np.abs(kept - 0.5)
    w_arr = np.maximum(w_arr, 1e-6)
    wmean_kept = float(np.average(kept, weights=w_arr))
    med_kept = float(np.median(kept))
    maj_frac = float((kept >= 0.5).mean()) if kept.size else 0.0

    return VideoDebugResult(
        prediction_label=label,
        prob_score=score,
        threshold_fake=threshold_fake,
        n_sampled=n_sampled,
        n_used_after_filter=n_used,
        prob_min=vmin,
        prob_max=vmax,
        prob_mean=mean,
        prob_std=std,
        prob_median=med_kept,
        prob_weighted_mean=wmean_kept,
        majority_fake_fraction=maj_frac,
        mean_prob=mean_prob,
        max_prob=max_prob,
        fake_frame_ratio=fake_frame_ratio,
        decision_reason=decision_reason,
    )


def load_runtime(
    arch: str,
    weights: str,
    meta_path: str,
    threshold_override: float,
    device_str: str,
) -> Tuple[nn.Module, transforms.Compose, torch.device, float, Dict[str, Any]]:
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    wpath = resolve_file(weights)
    mpath = resolve_file(meta_path)
    if not mpath.is_file():
        raise SystemExit(f"Meta file not found: {meta_path} (resolved: {mpath})")
    meta: Dict[str, Any] = json.loads(mpath.read_text(encoding="utf-8"))
    img_size = int(meta.get("img_size", 224))
    mean = meta.get("normalize_mean", [0.485, 0.456, 0.406])
    std = meta.get("normalize_std", [0.229, 0.224, 0.225])
    if threshold_override < 0:
        threshold_fake = 0.5  # Default to 0.5 instead of meta
    else:
        threshold_fake = float(threshold_override)

    mtcnn = MTCNN(keep_all=False, device='cpu')
    print("MTCNN running on CPU (fix for torchvision NMS issue)")
    tf = transforms.Compose(
        [
            FaceDetectionTransform(mtcnn, img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    model = build_model(arch).to(device)
    if not wpath.is_file():
        raise SystemExit(f"Weights not found: {weights} (resolved: {wpath})")
    model.load_state_dict(torch.load(wpath, map_location=device))
    model.eval()
    return model, tf, device, threshold_fake, meta


def print_video_debug(r: VideoDebugResult, video_name: str) -> None:
    print(f"Video: {video_name}")
    print(f"  Frames sampled: {r.n_sampled} | used after confidence filter: {r.n_used_after_filter}")
    print(f"  prob(fake) min:    {r.prob_min:.4f}")
    print(f"  prob(fake) max:    {r.prob_max:.4f}")
    print(f"  prob(fake) mean:   {r.prob_mean:.4f}")
    print(f"  prob(fake) std:    {r.prob_std:.4f}")
    print(f"  prob(fake) median (agg): {r.prob_median:.4f}")
    print(f"  prob(fake) w-mean (agg): {r.prob_weighted_mean:.4f}")
    print(f"  majority fake frac: {r.majority_fake_fraction:.3f}")
    print(f"  NEW METRICS:")
    print(f"    mean_prob: {r.mean_prob:.4f}")
    print(f"    max_prob: {r.max_prob:.4f}")
    print(f"    fake_frame_ratio: {r.fake_frame_ratio:.3f}")
    print(f"  Decision reason: {r.decision_reason}")
    print(f"  Aggregated score: {r.prob_score:.4f}  (threshold_fake={r.threshold_fake:.4f}) -> {r.prediction_label}")


def run_on_path(
    model: nn.Module,
    tf: transforms.Compose,
    device: torch.device,
    media_path: Path,
    n_frames: int,
    threshold_fake: float,
    meta: Dict[str, Any],
) -> None:
    p = Path(media_path)
    if not p.exists():
        print(f"Not found: {p}")
        return
    if p.is_dir():
        print(f"Expected a file, not a directory: {p}")
        return

    conf_margin = 0.0  # Disable aggressive filtering
    w_med = float(meta.get("video_median_weight", 0.55))
    w_w = float(meta.get("video_weighted_weight", 0.45))

    if p.suffix.lower() in VIDEO_EXTS:
        r = predict_video(
            model,
            tf,
            device,
            p,
            n_frames=n_frames,
            threshold_fake=threshold_fake,
            confidence_margin=conf_margin,
            w_median=w_med,
            w_weighted=w_w,
        )
        print_video_debug(r, str(p))
    else:
        # Image prediction
        img = Image.open(p).convert("RGB")
        img_tensor = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            prob_fake = float(probs[0, 0])
        pred = 0 if prob_fake >= threshold_fake else 1
        label = "FAKE" if pred == 0 else "REAL"
        print(f"Image: {p}")
        print(f"  prob(fake): {prob_fake:.4f}  (threshold_fake={threshold_fake:.4f}) -> {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, help="Path to image or video file")
    ap.add_argument("--arch", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    ap.add_argument("--weights", type=str, default="efficientnet_b0_best.pth")
    ap.add_argument("--meta", type=str, default="efficientnet_b0_meta.json")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--threshold_fake", type=float, default=-1.0, help="Override threshold from meta.json")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not args.path:
        # Interactive mode
        model, tf, device, threshold_fake, meta = load_runtime(
            args.arch, args.weights, args.meta, args.threshold_fake, args.device
        )
        print("Interactive mode. Enter paths (or 'q' to quit):")
        while True:
            try:
                path = input("Path: ").strip()
                if path.lower() in ("q", "quit", "exit"):
                    break
                run_on_path(model, tf, device, Path(path), args.frames, threshold_fake, meta)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # CLI mode
        model, tf, device, threshold_fake, meta = load_runtime(
            args.arch, args.weights, args.meta, args.threshold_fake, args.device
        )
        run_on_path(model, tf, device, Path(args.path), args.frames, threshold_fake, meta)


if __name__ == "__main__":
    main()