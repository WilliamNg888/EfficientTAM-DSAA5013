"""Run EfficientTAM on DAVIS2017 val and dump per-frame predicted masks.

Usage:
    python run_inference.py \\
        --ckpt /path/to/efficienttam_s.pt \\
        --cfg configs/efficienttam/efficienttam_s.yaml \\
        --davis /path/to/DAVIS \\
        --out /path/to/preds/efficienttam_s \\
        [--videos blackswan camel ...]   # default = all 30 val videos
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)


def load_first_frame_gt(annot_dir: Path) -> tuple[np.ndarray, list[int]]:
    """Read 00000.png (DAVIS palette PNG). Return (H,W) uint8 + list of obj ids."""
    first = sorted(annot_dir.glob("*.png"))[0]
    arr = np.array(Image.open(first))  # palette mode → values are object ids
    obj_ids = sorted(int(v) for v in np.unique(arr) if v != 0)
    return arr, obj_ids


def run_one_video(predictor, frames_dir: Path, annot_dir: Path, out_dir: Path) -> dict:
    """Propagate one video, save per-frame .npz (per-object bool masks)."""
    gt0, obj_ids = load_first_frame_gt(annot_dir)
    H, W = gt0.shape

    inference_state = predictor.init_state(video_path=str(frames_dir))

    for oid in obj_ids:
        obj_mask = (gt0 == oid)
        predictor.add_new_mask(inference_state, frame_idx=0, obj_id=oid, mask=obj_mask)

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    for fidx, oids_out, mask_logits in predictor.propagate_in_video(inference_state):
        masks = {}
        for i, oid in enumerate(oids_out):
            m = (mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.bool_)
            # Resize to original H,W if needed
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
            masks[int(oid)] = m
        # save: one file per frame, compact uint8 packed (obj_id_layer => bool)
        np.savez_compressed(out_dir / f"{fidx:05d}.npz", **{str(k): v for k, v in masks.items()})
        n_frames += 1

    elapsed = time.time() - t0
    predictor.reset_state(inference_state)
    return {
        "n_frames": n_frames,
        "n_objects": len(obj_ids),
        "obj_ids": obj_ids,
        "elapsed_s": elapsed,
        "fps": n_frames / elapsed if elapsed > 0 else 0,
        "HxW": [int(H), int(W)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True,
                    help="model_cfg name, e.g. configs/efficienttam/efficienttam_s.yaml")
    ap.add_argument("--davis", required=True, help="DAVIS root dir")
    ap.add_argument("--out", required=True, help="output dir for predicted masks")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="optional subset of video names; default = all val")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    davis = Path(args.davis)
    val_list = (davis / "ImageSets/2017/val.txt").read_text().split()
    if args.videos:
        val_list = [v for v in val_list if v in args.videos]
    print(f"[run] {len(val_list)} videos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device}, ckpt={args.ckpt}, cfg={args.cfg}")

    # bf16 autocast as in the example
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Disable torch.compile: triton's aarch64-conda-cc fails on GH200 even though sm_90 >= 8
    predictor = build_efficienttam_video_predictor(
        args.cfg, args.ckpt, device=device,
        hydra_overrides_extra=["++model.compile_image_encoder=False"],
    )

    summary = {}
    for v in val_list:
        frames_dir = davis / "JPEGImages/480p" / v
        annot_dir = davis / "Annotations/480p" / v
        out_dir = out_root / v
        if (out_dir / "_done.json").exists():
            print(f"[skip] {v} (already done)")
            with open(out_dir / "_done.json") as f:
                summary[v] = json.load(f)
            continue
        print(f"[run] {v} ...", flush=True)
        info = run_one_video(predictor, frames_dir, annot_dir, out_dir)
        with open(out_dir / "_done.json", "w") as f:
            json.dump(info, f)
        summary[v] = info
        print(f"  {info['n_frames']} frames, {info['n_objects']} obj, {info['fps']:.1f} fps", flush=True)

    total_frames = sum(s["n_frames"] for s in summary.values())
    total_time = sum(s["elapsed_s"] for s in summary.values())
    print(f"\n[done] {len(summary)} videos, {total_frames} frames, {total_frames/total_time:.1f} avg fps")
    with open(out_root / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
