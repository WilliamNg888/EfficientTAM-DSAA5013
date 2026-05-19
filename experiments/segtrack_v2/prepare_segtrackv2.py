"""Convert SegTrack v2 to the DAVIS-style layout used by our VOS runner.

Expected raw layout, based on the public SegTrack v2 release:

    <raw-root>/
      JPEGImages/<video>/<frame>.png
      GroundTruth/<video>/<frame>.png                 # single-object/indexed

or

    <raw-root>/
      JPEGImages/<video>/<frame>.png
      GroundTruth/<video>/<object-id>/<frame>.png     # multi-object binary masks

The converted output is:

    <output-root>/
      JPEGImages/480p/<video>/00000.jpg
      Annotations/480p/<video>/00000.png
      ImageSets/2017/val.txt

This keeps the EfficientTAM inference/evaluation protocol identical to DAVIS:
first-frame GT mask prompt, propagation, then J/F/J&F evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".png", ".bmp", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class VideoConversion:
    video: str
    num_frames: int
    num_objects: int
    status: str


def sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    if digits:
        return int(digits[-1]), path.name
    return sys.maxsize, path.name


def list_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=sort_key,
    )


def find_child(root: Path, names: list[str]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    lower_to_path = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for name in names:
        candidate = lower_to_path.get(name.lower())
        if candidate is not None:
            return candidate
    raise FileNotFoundError(f"Could not find any of {names} under {root}")


def frame_key(path: Path) -> str:
    return path.stem


def read_binary_mask(path: Path) -> np.ndarray:
    array = np.array(Image.open(path))
    if array.ndim == 2:
        return array > 0
    if array.shape[-1] == 4:
        return array[..., 3] > 0
    return np.any(array[..., :3] > 0, axis=-1)


def read_index_or_binary_mask(path: Path) -> np.ndarray:
    array = np.array(Image.open(path))
    if array.ndim == 2:
        labels = np.unique(array)
        labels = labels[labels != 0]
        if len(labels) > 1:
            return array.astype(np.uint16)
        return (array > 0).astype(np.uint16)
    binary = read_binary_mask(path)
    return binary.astype(np.uint16)


def write_index_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(mask.max()) if mask.size else 0
    if max_label <= 255:
        image = Image.fromarray(mask.astype(np.uint8), mode="P")
        palette = [0, 0, 0]
        colors = [
            (220, 20, 60),
            (0, 128, 255),
            (0, 180, 90),
            (255, 165, 0),
            (160, 80, 220),
            (255, 220, 0),
            (0, 200, 200),
            (180, 80, 80),
        ]
        for idx in range(1, 256):
            palette.extend(colors[(idx - 1) % len(colors)])
        image.putpalette(palette[: 256 * 3])
    else:
        image = Image.fromarray(mask.astype(np.uint16), mode="I;16")
    image.save(path)


def save_frame_as_jpeg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.save(dst, quality=95)


def match_masks_by_frame(
    frame_paths: list[Path],
    mask_paths: list[Path],
    object_name: str,
) -> dict[str, Path]:
    by_stem = {frame_key(path): path for path in mask_paths}
    frame_stems = [frame_key(path) for path in frame_paths]
    if all(stem in by_stem for stem in frame_stems):
        return {stem: by_stem[stem] for stem in frame_stems}
    if len(mask_paths) == len(frame_paths):
        return {frame_key(frame): mask for frame, mask in zip(frame_paths, mask_paths)}
    missing = [stem for stem in frame_stems if stem not in by_stem][:5]
    raise ValueError(
        f"Could not align masks for object '{object_name}'. "
        f"frames={len(frame_paths)}, masks={len(mask_paths)}, missing={missing}"
    )


def build_annotation_sequence(frame_paths: list[Path], gt_video_dir: Path) -> tuple[list[np.ndarray], int]:
    child_dirs = sorted([p for p in gt_video_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if child_dirs:
        object_maps = []
        for object_dir in child_dirs:
            mask_paths = list_files(object_dir, MASK_EXTS)
            if not mask_paths:
                continue
            object_maps.append((object_dir.name, match_masks_by_frame(frame_paths, mask_paths, object_dir.name)))
        if not object_maps:
            raise FileNotFoundError(f"No object masks found under {gt_video_dir}")

        annotations: list[np.ndarray] = []
        for frame in frame_paths:
            label_mask: np.ndarray | None = None
            stem = frame_key(frame)
            for obj_idx, (_, mask_map) in enumerate(object_maps, start=1):
                binary = read_binary_mask(mask_map[stem])
                if label_mask is None:
                    label_mask = np.zeros(binary.shape, dtype=np.uint16)
                label_mask[binary] = obj_idx
            assert label_mask is not None
            annotations.append(label_mask)
        return annotations, len(object_maps)

    mask_paths = list_files(gt_video_dir, MASK_EXTS)
    if not mask_paths:
        raise FileNotFoundError(f"No masks found under {gt_video_dir}")
    mask_map = match_masks_by_frame(frame_paths, mask_paths, gt_video_dir.name)
    annotations = [read_index_or_binary_mask(mask_map[frame_key(frame)]) for frame in frame_paths]
    object_count = int(max((int(mask.max()) for mask in annotations), default=0))
    return annotations, object_count


def convert_video(
    video: str,
    image_video_dir: Path,
    gt_video_dir: Path,
    output_root: Path,
    overwrite: bool,
) -> VideoConversion:
    frame_paths = list_files(image_video_dir, IMAGE_EXTS)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found under {image_video_dir}")

    out_image_dir = output_root / "JPEGImages" / "480p" / video
    out_ann_dir = output_root / "Annotations" / "480p" / video
    if out_image_dir.exists() and out_ann_dir.exists() and not overwrite:
        return VideoConversion(video, len(frame_paths), -1, "skipped_existing")

    if overwrite:
        shutil.rmtree(out_image_dir, ignore_errors=True)
        shutil.rmtree(out_ann_dir, ignore_errors=True)

    annotations, object_count = build_annotation_sequence(frame_paths, gt_video_dir)
    for idx, (frame_path, mask) in enumerate(zip(frame_paths, annotations)):
        save_frame_as_jpeg(frame_path, out_image_dir / f"{idx:05d}.jpg")
        write_index_mask(out_ann_dir / f"{idx:05d}.png", mask)

    return VideoConversion(video, len(frame_paths), object_count, "converted")


def write_manifest(output_root: Path, rows: list[VideoConversion]) -> None:
    split_dir = output_root / "ImageSets" / "2017"
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "val.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.video}\n")

    with (output_root / "conversion_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "num_frames", "num_objects", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    with (output_root / "conversion_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_videos": len(rows),
                "num_frames": sum(row.num_frames for row in rows),
                "num_objects": sum(max(row.num_objects, 0) for row in rows),
                "layout": "DAVIS-style: JPEGImages/480p, Annotations/480p, ImageSets/2017/val.txt",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Raw SegTrack v2 root containing JPEGImages and GroundTruth.")
    parser.add_argument("--output-root", required=True, help="Converted DAVIS-style output root.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_root = find_child(raw_root, ["JPEGImages", "Images", "frames", "Frames"])
    gt_root = find_child(raw_root, ["GroundTruth", "Annotations", "Groundtruth", "groundtruth"])

    videos = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    if not videos:
        raise FileNotFoundError(f"No video folders found under {image_root}")

    rows: list[VideoConversion] = []
    for video in videos:
        gt_video_dir = gt_root / video
        if not gt_video_dir.is_dir():
            raise FileNotFoundError(f"Missing GroundTruth folder for video '{video}': {gt_video_dir}")
        rows.append(convert_video(video, image_root / video, gt_video_dir, output_root, args.overwrite))
        row = rows[-1]
        print(f"{row.status}: {row.video} ({row.num_frames} frames, {row.num_objects} objects)")

    write_manifest(output_root, rows)
    print(f"\nConverted SegTrack v2 to: {output_root}")
    print(f"Split file: {output_root / 'ImageSets' / '2017' / 'val.txt'}")


if __name__ == "__main__":
    main()
