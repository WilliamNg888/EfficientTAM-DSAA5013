"""Validate and summarize SegTrack v2 EfficientTAM inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def fmt(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-videos", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted_root = Path(args.converted_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    manifest = load_json(converted_root / "conversion_manifest.json")
    s_summary = load_json(output_root / "eval_s" / "summary.json")
    s2_summary = load_json(output_root / "eval_s2" / "summary.json")
    s_rows = count_rows(output_root / "eval_s" / "per_video.csv")
    s2_rows = count_rows(output_root / "eval_s2" / "per_video.csv")

    errors = []
    if manifest.get("num_videos") != args.expected_videos:
        errors.append(
            f"expected {args.expected_videos} converted videos, got {manifest.get('num_videos')}"
        )
    if s_summary.get("num_videos") != manifest.get("num_videos"):
        errors.append("EfficientTAM-S evaluated video count does not match conversion manifest")
    if s2_summary.get("num_videos") != manifest.get("num_videos"):
        errors.append("EfficientTAM-S/2 evaluated video count does not match conversion manifest")
    if s_rows != manifest.get("num_videos"):
        errors.append("eval_s/per_video.csv row count does not match conversion manifest")
    if s2_rows != manifest.get("num_videos"):
        errors.append("eval_s2/per_video.csv row count does not match conversion manifest")

    print("SegTrack v2 result check")
    print("========================")
    print(f"Converted videos: {manifest.get('num_videos')}")
    print(f"Converted frames: {manifest.get('num_frames')}")
    print(f"Converted objects: {manifest.get('num_objects')}")
    print()
    print("| Dataset | Model | J | F | J&F |")
    print("|---|---|---:|---:|---:|")
    print(
        f"| SegTrack v2 | EfficientTAM-S | {fmt(s_summary['mean_j'])} | "
        f"{fmt(s_summary['mean_f'])} | {fmt(s_summary['mean_jf'])} |"
    )
    print(
        f"| SegTrack v2 | EfficientTAM-S/2 | {fmt(s2_summary['mean_j'])} | "
        f"{fmt(s2_summary['mean_f'])} | {fmt(s2_summary['mean_jf'])} |"
    )

    if errors:
        print("\nFAILED checks:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nAll required result files are present and internally consistent.")


if __name__ == "__main__":
    main()
