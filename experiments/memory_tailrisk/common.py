"""Shared utilities for EfficientTAM memory tail-risk experiments.

The helpers in this folder intentionally avoid importing private project code
except in the runner script.  This keeps metric/risk analysis usable on saved
mask outputs without a GPU or EfficientTAM environment.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class VideoRecord:
    name: str
    image_dir: Path
    annotation_dir: Path | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def davis_image_root(dataset_root: Path, resolution: str = "480p") -> Path:
    candidates = [
        dataset_root / "JPEGImages" / resolution,
        dataset_root / "JPEGImages",
        dataset_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find DAVIS image root under {dataset_root}")


def davis_annotation_root(dataset_root: Path, resolution: str = "480p") -> Path:
    candidates = [
        dataset_root / "Annotations" / resolution,
        dataset_root / "Annotations",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find DAVIS annotation root under {dataset_root}")


def read_split(split_file: Path | None) -> list[str] | None:
    if split_file is None:
        return None
    names: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split()[0])
    return names


def list_davis_videos(
    dataset_root: Path,
    split_file: Path | None = None,
    resolution: str = "480p",
    max_videos: int | None = None,
) -> list[VideoRecord]:
    image_root = davis_image_root(dataset_root, resolution)
    annotation_root = davis_annotation_root(dataset_root, resolution)
    names = read_split(split_file)
    if names is None:
        names = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    if max_videos is not None:
        names = names[:max_videos]
    records: list[VideoRecord] = []
    for name in names:
        image_dir = image_root / name
        annotation_dir = annotation_root / name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory for {name}: {image_dir}")
        if not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Missing annotation directory for {name}: {annotation_dir}"
            )
        records.append(VideoRecord(name=name, image_dir=image_dir, annotation_dir=annotation_dir))
    return records


def sorted_frame_paths(frame_dir: Path, extensions: set[str] = IMAGE_EXTENSIONS) -> list[Path]:
    frames = [p for p in frame_dir.iterdir() if p.suffix.lower() in extensions]
    if not frames:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    def key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return math.inf, path.name

    return sorted(frames, key=key)


def read_index_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def read_palette(path: Path) -> list[int] | None:
    palette = Image.open(path).getpalette()
    return palette if palette else None


def write_index_mask(path: Path, mask: np.ndarray, palette: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(mask.max()) if mask.size else 0
    if max_label <= 255:
        image = Image.fromarray(mask.astype(np.uint8), mode="P")
        if palette is not None:
            image.putpalette(palette)
    else:
        image = Image.fromarray(mask.astype(np.uint16), mode="I;16")
    image.save(path)


def mask_labels(mask: np.ndarray) -> list[int]:
    labels = np.unique(mask)
    return [int(x) for x in labels if int(x) != 0]


def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def _disk_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def boundary_f_measure(
    pred: np.ndarray,
    target: np.ndarray,
    dilation_ratio: float = 0.008,
) -> float:
    from scipy import ndimage

    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    if not pred.any() or not target.any():
        return 0.0

    pred_boundary = mask_boundary(pred)
    target_boundary = mask_boundary(target)
    diag = math.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    radius = max(1, int(round(dilation_ratio * diag)))
    structure = _disk_structure(radius)
    pred_match = ndimage.binary_dilation(pred_boundary, structure=structure)
    target_match = ndimage.binary_dilation(target_boundary, structure=structure)

    pred_count = pred_boundary.sum()
    target_count = target_boundary.sum()
    precision = (
        np.logical_and(pred_boundary, target_match).sum() / pred_count
        if pred_count
        else 0.0
    )
    recall = (
        np.logical_and(target_boundary, pred_match).sum() / target_count
        if target_count
        else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def connected_components(mask: np.ndarray) -> int:
    from scipy import ndimage

    if not mask.any():
        return 0
    _, count = ndimage.label(mask.astype(bool))
    return int(count)


def top_fraction_mean(values: Iterable[float], fraction: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(array.size * fraction)))
    return float(np.sort(array)[-k:].mean())


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x_arr, y_arr).statistic
        return float(value)
    except Exception:
        x_rank = rankdata(x_arr)
        y_rank = rankdata(y_arr)
        return float(np.corrcoef(x_rank, y_rank)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks
