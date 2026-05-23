"""Compute per-video J / F / J&F by comparing predicted .npz masks vs DAVIS GT.

Standard DAVIS metric:
  J = mean IoU(pred, gt) across frames (per object, skip frame 0)
  F = boundary F-measure (bound_pix = 0.008 * sqrt(H^2+W^2))
  J&F = (J + F) / 2

Per video, average over all annotated objects (DAVIS-style "per-object then per-video").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt


# ---------- F boundary measure (DAVIS official spec) ----------
def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Binary boundary map from a binary segmentation (DAVIS official, 4-neighborhood)."""
    seg = seg.astype(bool)
    h, w = seg.shape
    e = np.zeros_like(seg); s = np.zeros_like(seg); se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_pix: int | None = None) -> float:
    """Boundary F-measure, dilation tolerance bound_pix (default = 0.008 * diag)."""
    if bound_pix is None:
        h, w = pred.shape
        bound_pix = int(np.ceil(0.008 * np.sqrt(h * h + w * w)))
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    fg_b = _seg2bmap(pred)
    gt_b = _seg2bmap(gt)
    # dilate by bound_pix
    struct = np.ones((2 * bound_pix + 1, 2 * bound_pix + 1), dtype=bool)
    fg_b_d = binary_dilation(fg_b, structure=struct)
    gt_b_d = binary_dilation(gt_b, structure=struct)
    gt_match = (gt_b & fg_b_d).sum()
    fg_match = (fg_b & gt_b_d).sum()
    n_fg = fg_b.sum()
    n_gt = gt_b.sum()
    if n_fg == 0 and n_gt > 0:
        return 0.0
    if n_gt == 0 and n_fg > 0:
        return 0.0
    P = fg_match / max(n_fg, 1)
    R = gt_match / max(n_gt, 1)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    uni = (p | g).sum()
    if uni == 0:
        return 1.0
    return inter / uni


def load_pred(npz_path: Path) -> dict[int, np.ndarray]:
    z = np.load(npz_path)
    return {int(k): z[k].astype(bool) for k in z.files}


def eval_video(pred_dir: Path, annot_dir: Path) -> dict:
    """Return per-object J/F arrays and means (skipping frame 0)."""
    annot_files = sorted(annot_dir.glob("*.png"))
    gt0 = np.array(Image.open(annot_files[0]))
    obj_ids = sorted(int(v) for v in np.unique(gt0) if v != 0)

    per_obj_j = {oid: [] for oid in obj_ids}
    per_obj_f = {oid: [] for oid in obj_ids}

    for fidx, annot_file in enumerate(annot_files):
        if fidx == 0:
            continue  # skip prompted frame per DAVIS convention
        gt = np.array(Image.open(annot_file))
        pred_path = pred_dir / f"{fidx:05d}.npz"
        if not pred_path.exists():
            # missing prediction → treat as empty
            preds = {}
        else:
            preds = load_pred(pred_path)
        for oid in obj_ids:
            gt_o = (gt == oid)
            pred_o = preds.get(oid, np.zeros_like(gt_o))
            if pred_o.shape != gt_o.shape:
                # resize pred to gt shape (NEAREST)
                pred_o = np.array(Image.fromarray(pred_o.astype(np.uint8) * 255)
                                  .resize((gt_o.shape[1], gt_o.shape[0]), Image.NEAREST)) > 127
            per_obj_j[oid].append(iou(pred_o, gt_o))
            per_obj_f[oid].append(boundary_f(pred_o, gt_o))

    # per-video J = mean over objects of (mean over frames)
    obj_J = {oid: float(np.mean(per_obj_j[oid])) if per_obj_j[oid] else 0.0 for oid in obj_ids}
    obj_F = {oid: float(np.mean(per_obj_f[oid])) if per_obj_f[oid] else 0.0 for oid in obj_ids}
    J_video = float(np.mean(list(obj_J.values())))
    F_video = float(np.mean(list(obj_F.values())))
    return {
        "J": J_video,
        "F": F_video,
        "JF": (J_video + F_video) / 2,
        "per_object_J": obj_J,
        "per_object_F": obj_F,
        "obj_ids": obj_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir with per-video subdirs of .npz")
    ap.add_argument("--davis", required=True, help="DAVIS root")
    ap.add_argument("--out", required=True, help="output json")
    args = ap.parse_args()

    davis = Path(args.davis)
    pred_root = Path(args.pred)
    val = (davis / "ImageSets/2017/val.txt").read_text().split()

    out = {}
    for v in val:
        pred_dir = pred_root / v
        annot_dir = davis / "Annotations/480p" / v
        if not pred_dir.exists():
            print(f"[skip] {v}: no pred dir")
            continue
        r = eval_video(pred_dir, annot_dir)
        out[v] = r
        print(f"{v}: J={r['J']*100:.2f}  F={r['F']*100:.2f}  J&F={r['JF']*100:.2f}", flush=True)

    if out:
        meanJ = np.mean([r["J"] for r in out.values()]) * 100
        meanF = np.mean([r["F"] for r in out.values()]) * 100
        meanJF = np.mean([r["JF"] for r in out.values()]) * 100
        print(f"\n=== MEAN  J={meanJ:.2f}  F={meanF:.2f}  J&F={meanJF:.2f} ===")
        out["__mean__"] = {"J": meanJ / 100, "F": meanF / 100, "JF": meanJF / 100, "n_videos": len(out) - 0}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
