"""Evaluate indexed-mask VOS predictions with per-video J, F, and J&F."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    binary_iou,
    boundary_f_measure,
    list_davis_videos,
    mask_labels,
    read_index_mask,
    resolve_path,
    sorted_frame_paths,
    write_csv_dicts,
    write_json,
)


def evaluate_video(video, pred_dir: Path, skip_first_frame: bool) -> tuple[list[dict], dict]:
    assert video.annotation_dir is not None
    gt_paths = sorted_frame_paths(video.annotation_dir, extensions={".png"})
    first_mask = read_index_mask(gt_paths[0])
    obj_ids = mask_labels(first_mask)
    if not obj_ids:
        obj_ids = sorted({label for path in gt_paths for label in mask_labels(read_index_mask(path))})

    per_object_rows = []
    video_j_values = []
    video_f_values = []
    frame_indices = range(1 if skip_first_frame else 0, len(gt_paths))
    for obj_id in obj_ids:
        j_values = []
        f_values = []
        for frame_idx in frame_indices:
            gt_path = gt_paths[frame_idx]
            pred_path = pred_dir / video.name / gt_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")
            gt_mask = read_index_mask(gt_path) == obj_id
            pred_mask = read_index_mask(pred_path) == obj_id
            j_values.append(binary_iou(pred_mask, gt_mask))
            f_values.append(boundary_f_measure(pred_mask, gt_mask))
        j_mean = float(np.mean(j_values)) if j_values else float("nan")
        f_mean = float(np.mean(f_values)) if f_values else float("nan")
        jf = 0.5 * (j_mean + f_mean)
        per_object_rows.append(
            {
                "video": video.name,
                "object_id": obj_id,
                "j": j_mean,
                "f": f_mean,
                "jf": jf,
                "num_eval_frames": len(j_values),
            }
        )
        video_j_values.append(j_mean)
        video_f_values.append(f_mean)

    video_j = float(np.mean(video_j_values)) if video_j_values else float("nan")
    video_f = float(np.mean(video_f_values)) if video_f_values else float("nan")
    per_video = {
        "video": video.name,
        "num_objects": len(obj_ids),
        "j": video_j,
        "f": video_f,
        "jf": 0.5 * (video_j + video_f),
    }
    return per_object_rows, per_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--include-first-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    pred_dir = resolve_path(args.pred_dir)
    output_dir = resolve_path(args.output_dir)
    split_file = resolve_path(args.split_file) if args.split_file else None
    videos = list_davis_videos(dataset_root, split_file, args.resolution, args.max_videos)

    object_rows: list[dict] = []
    video_rows: list[dict] = []
    for video in tqdm(videos, desc="evaluating J&F"):
        per_object, per_video = evaluate_video(
            video, pred_dir, skip_first_frame=not args.include_first_frame
        )
        object_rows.extend(per_object)
        video_rows.append(per_video)

    write_csv_dicts(
        output_dir / "per_object.csv",
        object_rows,
        ["video", "object_id", "j", "f", "jf", "num_eval_frames"],
    )
    write_csv_dicts(output_dir / "per_video.csv", video_rows, ["video", "num_objects", "j", "f", "jf"])
    summary = {
        "num_videos": len(video_rows),
        "mean_j": float(np.mean([r["j"] for r in video_rows])),
        "mean_f": float(np.mean([r["f"] for r in video_rows])),
        "mean_jf": float(np.mean([r["jf"] for r in video_rows])),
        "skip_first_frame": not args.include_first_frame,
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
