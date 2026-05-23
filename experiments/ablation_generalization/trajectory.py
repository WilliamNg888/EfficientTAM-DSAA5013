"""Per-frame J trajectory for selected hard videos across the 6 ckpts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "outputs/ablation_generalization"
FIGS = RES / "figs"

CKPTS = [
    ("efficienttam_s",         "S",        "#1f77b4", "-"),
    ("efficienttam_s_2",       "S/2",      "#1f77b4", "--"),
    ("efficienttam_s_1",       "S/1",      "#1f77b4", ":"),
    ("efficienttam_ti",        "Ti",       "#ff7f0e", "-"),
    ("efficienttam_ti_2",      "Ti/2",     "#ff7f0e", "--"),
    ("efficienttam_s_512x512", "S$_{512}$","#2ca02c", "-"),
]

TARGET_VIDEOS = ["kite-surf", "lab-coat", "paragliding-launch", "bmx-trees"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    if u == 0:
        return 1.0
    return (a & b).sum() / u


def load_pred(npz_path: Path) -> dict[int, np.ndarray]:
    z = np.load(npz_path)
    return {int(k): z[k].astype(bool) for k in z.files}


def per_frame_J(davis: Path, preds_root: Path, video: str, ckpt: str):
    annot_dir = davis / "Annotations/480p" / video
    pred_dir = preds_root / ckpt / video
    annot_files = sorted(annot_dir.glob("*.png"))
    gt0 = np.array(Image.open(annot_files[0]))
    obj_ids = sorted(int(v) for v in np.unique(gt0) if v != 0)
    per_obj = {oid: [] for oid in obj_ids}
    frame_idx_list = []
    for fi, ann in enumerate(annot_files):
        if fi == 0:
            continue
        gt = np.array(Image.open(ann))
        pred_path = pred_dir / f"{fi:05d}.npz"
        preds = load_pred(pred_path) if pred_path.exists() else {}
        for oid in obj_ids:
            gt_o = (gt == oid)
            pred_o = preds.get(oid, np.zeros_like(gt_o))
            if pred_o.shape != gt_o.shape:
                pred_o = np.array(
                    Image.fromarray(pred_o.astype(np.uint8) * 255)
                    .resize((gt_o.shape[1], gt_o.shape[0]), Image.NEAREST)
                ) > 127
            per_obj[oid].append(iou(pred_o, gt_o))
        frame_idx_list.append(fi)
    return frame_idx_list, per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", required=True, help="DAVIS root")
    ap.add_argument("--preds-root", required=True, help="dir with <ckpt>/<video>/<frame>.npz")
    args = ap.parse_args()
    davis = Path(args.davis)
    preds_root = Path(args.preds_root)

    out = {}
    for v in TARGET_VIDEOS:
        print(f"\n=== {v} ===")
        annot0 = sorted((davis / "Annotations/480p" / v).glob("*.png"))[0]
        gt0 = np.array(Image.open(annot0))
        obj_ids = sorted(int(x) for x in np.unique(gt0) if x != 0)
        n_obj = len(obj_ids)

        traj = {}
        for ck, label, color, ls in CKPTS:
            idx, per_obj = per_frame_J(davis, preds_root, v, ck)
            arr = np.array([per_obj[oid] for oid in obj_ids])
            mean_J = arr.mean(axis=0)
            traj[label] = {"idx": idx, "J": mean_J.tolist()}
            print(f"  {label}: mean J = {mean_J.mean()*100:.2f}, min J = {mean_J.min()*100:.2f}")
        out[v] = {"n_objects": n_obj, "obj_ids": obj_ids, "traj": traj}

        plt.figure(figsize=(9, 4.5))
        for ck, label, color, ls in CKPTS:
            d = traj[label]
            plt.plot(d["idx"], np.array(d["J"]) * 100, color=color, linestyle=ls,
                     label=label, linewidth=1.8, alpha=0.85)
        plt.xlabel("Frame index (frame 0 = prompt, excluded)")
        plt.ylabel("Per-frame J (%)")
        plt.legend(fontsize=8, ncol=3, loc="lower left")
        plt.grid(alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(FIGS / f"trajectory_{v}.png", dpi=150)
        plt.close()
        print(f"  saved -> {FIGS / f'trajectory_{v}.png'}")

    with open(RES / "trajectory.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RES / 'trajectory.json'}")


if __name__ == "__main__":
    main()
