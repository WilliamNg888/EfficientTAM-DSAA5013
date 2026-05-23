"""Compute mask-only risk score per video (no GT needed).

Per the formulas in paper/DSAA_5013/old.tex section 4.1:
  r_area(t)  = |log((|M_t|+eps) / (|M_{t-1}|+eps))|
  r_iou(t)   = 1 - IoU(M_t, M_{t-1})
  r_frag(t)  = #connected_components(M_t)

For each video:
  per-object, per-frame compute the three signals
  aggregate per metric: mean of top-20% frame-level values
  → per-video three numbers
  robust-z normalize each metric across videos (median / MAD)
  average → final R(v)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

EPS = 1.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    if u == 0:
        return 1.0
    return (a & b).sum() / u


def per_video_signals(pred_dir: Path, obj_ids: list[int]) -> dict[str, float]:
    """Return three aggregated metrics (mean of top-20% frame values)."""
    frame_files = sorted(pred_dir.glob("[0-9]*.npz"))
    per_frame_area = []   # list of r_area
    per_frame_iou = []
    per_frame_frag = []

    prev_masks: dict[int, np.ndarray] = {}
    for fi, fp in enumerate(frame_files):
        z = np.load(fp)
        masks = {int(k): z[k].astype(bool) for k in z.files}

        for oid in obj_ids:
            m = masks.get(oid)
            if m is None:
                continue
            # fragmentation: # connected components of mask
            if m.any():
                _, n_cc = label(m)
            else:
                n_cc = 0
            per_frame_frag.append(float(n_cc))

            prev = prev_masks.get(oid)
            if prev is not None:
                # area jump
                size_t = m.sum()
                size_p = prev.sum()
                r_a = abs(np.log((size_t + EPS) / (size_p + EPS)))
                per_frame_area.append(float(r_a))
                # IoU instability
                per_frame_iou.append(1.0 - iou(m, prev))
            prev_masks[oid] = m

    def topk_mean(xs: list[float], q: float = 0.2) -> float:
        if not xs:
            return 0.0
        arr = np.array(xs)
        k = max(1, int(np.ceil(q * len(arr))))
        return float(np.mean(np.sort(arr)[-k:]))

    return {
        "r_area_top20": topk_mean(per_frame_area),
        "r_iou_top20": topk_mean(per_frame_iou),
        "r_frag_top20": topk_mean(per_frame_frag),
        "n_frames": len(frame_files),
        "n_obj_pairs": len(per_frame_area),
    }


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        mad = np.std(x) + 1e-9
    return (x - med) / (1.4826 * mad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--davis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image

    davis = Path(args.davis)
    pred_root = Path(args.pred)
    val = (davis / "ImageSets/2017/val.txt").read_text().split()

    # Get obj_ids per video from first GT frame
    per_video: dict[str, dict] = {}
    for v in val:
        annot_dir = davis / "Annotations/480p" / v
        gt0 = np.array(Image.open(sorted(annot_dir.glob("*.png"))[0]))
        obj_ids = sorted(int(x) for x in np.unique(gt0) if x != 0)
        pred_dir = pred_root / v
        if not pred_dir.exists():
            continue
        per_video[v] = per_video_signals(pred_dir, obj_ids)

    # Robust z normalize across videos
    names = list(per_video.keys())
    area = np.array([per_video[n]["r_area_top20"] for n in names])
    iou_ = np.array([per_video[n]["r_iou_top20"] for n in names])
    frag = np.array([per_video[n]["r_frag_top20"] for n in names])

    z_area = robust_z(area)
    z_iou = robust_z(iou_)
    z_frag = robust_z(frag)
    R = (z_area + z_iou + z_frag) / 3

    for i, n in enumerate(names):
        per_video[n]["z_area"] = float(z_area[i])
        per_video[n]["z_iou"] = float(z_iou[i])
        per_video[n]["z_frag"] = float(z_frag[i])
        per_video[n]["R"] = float(R[i])

    # rank
    order = np.argsort(-R)
    top20_n = max(1, int(np.ceil(0.2 * len(names))))
    top_set = [names[i] for i in order[:top20_n]]
    print(f"Top-{top20_n} highest-risk videos: {top_set}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_video": per_video, "top_risk": top_set}, f, indent=2)


if __name__ == "__main__":
    main()
