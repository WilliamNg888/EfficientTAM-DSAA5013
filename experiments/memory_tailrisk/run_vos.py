"""Run EfficientTAM VOS inference on a DAVIS-style dataset.

This script uses the first annotation frame as mask prompts, propagates masks
through the video, and saves indexed PNG predictions.  It does not modify the
upstream EfficientTAM package.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    list_davis_videos,
    mask_labels,
    read_index_mask,
    read_palette,
    repo_root,
    resolve_path,
    sorted_frame_paths,
    write_csv_dicts,
    write_index_mask,
    write_json,
)


MODEL_PRESETS = {
    "s": ("configs/efficienttam/efficienttam_s.yaml", "checkpoints/efficienttam_s.pt"),
    "s2": ("configs/efficienttam/efficienttam_s_2.yaml", "checkpoints/efficienttam_s_2.pt"),
}


def add_repo_to_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def load_predictor(args: argparse.Namespace):
    add_repo_to_path()
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

    if args.model_preset:
        config, checkpoint = MODEL_PRESETS[args.model_preset]
    else:
        config, checkpoint = args.config, args.checkpoint

    config_path = str(config)
    checkpoint_path = resolve_path(checkpoint) if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        if not args.allow_random_weights:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run checkpoints/download_checkpoints.sh "
                "or pass --checkpoint. Use --allow-random-weights only for a smoke test."
            )
        checkpoint_path = None

    hydra_overrides = []
    if args.non_overlap_masks:
        hydra_overrides.append("++model.non_overlap_masks=true")

    device = torch.device(args.device) if args.device else default_device()
    configure_torch(device)
    return build_efficienttam_video_predictor(
        config_path,
        str(checkpoint_path) if checkpoint_path is not None else None,
        device=device,
        vos_optimized=args.vos_optimized,
        hydra_overrides_extra=hydra_overrides,
    )


def logits_to_index_mask(logits: torch.Tensor, obj_ids: list[int]) -> np.ndarray:
    scores = logits.detach().float().cpu().numpy()
    if scores.ndim == 4:
        scores = scores[:, 0]
    max_index = np.argmax(scores, axis=0)
    max_score = np.max(scores, axis=0)
    output = np.zeros(max_score.shape, dtype=np.uint16)
    labels = np.asarray(obj_ids, dtype=np.uint16)
    output[max_score > 0] = labels[max_index[max_score > 0]]
    return output


def run_video(predictor, video, output_root: Path, args: argparse.Namespace) -> dict:
    assert video.annotation_dir is not None
    frame_paths = sorted_frame_paths(video.image_dir)
    annotation_paths = sorted_frame_paths(video.annotation_dir, extensions={".png"})
    first_mask = read_index_mask(annotation_paths[0])
    obj_ids = mask_labels(first_mask)
    if not obj_ids:
        raise ValueError(f"No foreground labels in first annotation for {video.name}")
    palette = read_palette(annotation_paths[0])

    video_output_dir = output_root / video.name
    if video_output_dir.exists() and not args.overwrite:
        existing = list(video_output_dir.glob("*.png"))
        if len(existing) >= len(frame_paths):
            return {
                "video": video.name,
                "num_frames": len(frame_paths),
                "num_objects": len(obj_ids),
                "seconds": 0.0,
                "fps": float("nan"),
                "status": "skipped_existing",
            }

    start = time.perf_counter()
    inference_state = predictor.init_state(
        video_path=str(video.image_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    for obj_id in obj_ids:
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=int(obj_id),
            mask=(first_mask == obj_id),
        )

    saved = 0
    with torch.inference_mode():
        iterator = predictor.propagate_in_video(inference_state)
        for frame_idx, out_obj_ids, out_mask_logits in iterator:
            output_mask = logits_to_index_mask(out_mask_logits, [int(x) for x in out_obj_ids])
            out_name = annotation_paths[frame_idx].name if frame_idx < len(annotation_paths) else f"{frame_idx:05d}.png"
            write_index_mask(video_output_dir / out_name, output_mask, palette)
            saved += 1

    seconds = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "video": video.name,
        "num_frames": len(frame_paths),
        "num_objects": len(obj_ids),
        "seconds": seconds,
        "fps": len(frame_paths) / seconds if seconds > 0 else float("nan"),
        "status": f"saved_{saved}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="DAVIS root with JPEGImages/ and Annotations/.")
    parser.add_argument("--split-file", default=None, help="Optional val.txt-style file.")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="s2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or omitted for auto.")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video", action="append", default=None, help="Restrict to one or more video names.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-random-weights", action="store_true")
    parser.add_argument("--non-overlap-masks", action="store_true")
    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_preset and not (args.config and args.checkpoint):
        raise ValueError("Pass --model-preset or both --config and --checkpoint.")

    dataset_root = resolve_path(args.dataset_root)
    split_file = resolve_path(args.split_file) if args.split_file else None
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_davis_videos(dataset_root, split_file, args.resolution, args.max_videos)
    if args.video:
        requested = set(args.video)
        videos = [video for video in videos if video.name in requested]
    if not videos:
        raise ValueError("No videos selected.")

    predictor = load_predictor(args)
    rows = []
    for video in tqdm(videos, desc=f"running EfficientTAM-{args.model_preset}"):
        rows.append(run_video(predictor, video, output_dir, args))

    fieldnames = ["video", "num_frames", "num_objects", "seconds", "fps", "status"]
    write_csv_dicts(output_dir / "run_manifest.csv", rows, fieldnames)
    write_json(
        output_dir / "run_manifest.json",
        {
            "model_preset": args.model_preset,
            "dataset_root": str(dataset_root),
            "num_videos": len(rows),
            "mean_fps": float(np.nanmean([float(r["fps"]) for r in rows])),
        },
    )


if __name__ == "__main__":
    main()
