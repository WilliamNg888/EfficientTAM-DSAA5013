"""Compute mask-only temporal risk scores from S/2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    binary_iou,
    connected_components,
    mask_labels,
    read_index_mask,
    resolve_path,
    sorted_frame_paths,
    top_fraction_mean,
    write_csv_dicts,
    write_json,
)


def video_risk(pred_video_dir: Path, top_fraction: float, epsilon: float) -> dict:
    mask_paths = sorted_frame_paths(pred_video_dir, extensions={".png"})
    masks = [read_index_mask(path) for path in mask_paths]
    obj_ids = sorted({label for mask in masks for label in mask_labels(mask)})
    if not obj_ids:
        return {
            "video": pred_video_dir.name,
            "num_frames": len(masks),
            "num_objects": 0,
            "area_risk": 0.0,
            "iou_risk": 0.0,
            "frag_risk": 0.0,
        }

    area_series = []
    iou_series = []
    frag_series = []
    for t in range(1, len(masks)):
        per_obj_area = []
        per_obj_iou = []
        per_obj_frag = []
        prev = masks[t - 1]
        curr = masks[t]
        for obj_id in obj_ids:
            prev_obj = prev == obj_id
            curr_obj = curr == obj_id
            prev_area = float(prev_obj.sum())
            curr_area = float(curr_obj.sum())
            per_obj_area.append(abs(np.log((curr_area + epsilon) / (prev_area + epsilon))))
            per_obj_iou.append(1.0 - binary_iou(curr_obj, prev_obj))
            per_obj_frag.append(float(connected_components(curr_obj)))
        area_series.append(max(per_obj_area))
        iou_series.append(max(per_obj_iou))
        frag_series.append(max(per_obj_frag))

    return {
        "video": pred_video_dir.name,
        "num_frames": len(masks),
        "num_objects": len(obj_ids),
        "area_risk": top_fraction_mean(area_series, top_fraction),
        "iou_risk": top_fraction_mean(iou_series, top_fraction),
        "frag_risk": top_fraction_mean(frag_series, top_fraction),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    scale = q3 - q1
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - median) / scale


def add_composite_score(rows: list[dict]) -> None:
    if not rows:
        return
    metric_names = ["area_risk", "iou_risk", "frag_risk"]
    z_scores = []
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        z_scores.append(robust_z(values))
    composite = np.mean(np.vstack(z_scores), axis=0)
    for row, score in zip(rows, composite):
        row["risk_score"] = float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="S/2 prediction root containing one folder per video.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = resolve_path(args.pred_dir)
    output_dir = resolve_path(args.output_dir)
    video_dirs = sorted(p for p in pred_dir.iterdir() if p.is_dir())
    rows = [video_risk(path, args.top_fraction, args.epsilon) for path in tqdm(video_dirs, desc="risk")]
    add_composite_score(rows)
    rows.sort(key=lambda row: float(row["risk_score"]), reverse=True)
    fieldnames = [
        "video",
        "num_frames",
        "num_objects",
        "area_risk",
        "iou_risk",
        "frag_risk",
        "risk_score",
    ]
    write_csv_dicts(output_dir / "risk_scores.csv", rows, fieldnames)
    write_json(
        output_dir / "summary.json",
        {
            "num_videos": len(rows),
            "top_fraction": args.top_fraction,
            "score": "mean robust-z(area_risk, iou_risk, frag_risk)",
        },
    )


if __name__ == "__main__":
    main()
