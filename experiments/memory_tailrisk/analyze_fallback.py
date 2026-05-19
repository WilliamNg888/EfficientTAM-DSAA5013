"""Analyze S vs S/2 tail risk and risk-gated fallback."""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import read_csv_dicts, resolve_path, spearman_corr, top_fraction_mean, write_csv_dicts, write_json


def load_metric(path, value_key: str) -> dict[str, float]:
    return {row["video"]: float(row[value_key]) for row in read_csv_dicts(path)}


def write_cdf_svg(cdf_rows: list[dict], output_path) -> None:
    if not cdf_rows:
        return
    drops = [float(row["drop"]) for row in cdf_rows]
    cdf = [float(row["cdf"]) for row in cdf_rows]
    width, height = 560, 360
    left, right, top, bottom = 70, 25, 25, 55
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = min(drops), max(drops)
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0

    def xy(drop: float, value: float) -> tuple[float, float]:
        x = left + (drop - x_min) / (x_max - x_min) * plot_w
        y = top + (1.0 - value) * plot_h
        return x, y

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(d, p) for d, p in zip(drops, cdf)))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>
  <polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <text x="{left + plot_w / 2:.0f}" y="{height - 15}" text-anchor="middle" font-family="Arial" font-size="14">Per-video J&amp;F drop: S - S/2</text>
  <text x="18" y="{top + plot_h / 2:.0f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {top + plot_h / 2:.0f})">CDF</text>
  <text x="{left}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x_min:.3f}</text>
  <text x="{left + plot_w}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x_max:.3f}</text>
  <text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">0</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">1</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-per-video", required=True, help="evaluate_jf.py per_video.csv for EfficientTAM-S.")
    parser.add_argument("--s2-per-video", required=True, help="evaluate_jf.py per_video.csv for EfficientTAM-S/2.")
    parser.add_argument("--risk-scores", required=True, help="risk_score.py risk_scores.csv from S/2 predictions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.2)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    s = load_metric(resolve_path(args.s_per_video), "jf")
    s2 = load_metric(resolve_path(args.s2_per_video), "jf")
    risk = load_metric(resolve_path(args.risk_scores), "risk_score")

    videos = sorted(set(s) & set(s2) & set(risk))
    if not videos:
        raise ValueError("No overlapping videos across S, S/2, and risk files.")
    k = max(1, int(math.ceil(len(videos) * args.budget_fraction)))

    rows = []
    for video in videos:
        rows.append(
            {
                "video": video,
                "s_jf": s[video],
                "s2_jf": s2[video],
                "drop": s[video] - s2[video],
                "risk_score": risk[video],
            }
        )
    rows.sort(key=lambda row: row["drop"], reverse=True)
    write_csv_dicts(output_dir / "per_video_delta.csv", rows, ["video", "s_jf", "s2_jf", "drop", "risk_score"])

    cdf_rows = []
    sorted_drops = sorted(row["drop"] for row in rows)
    for idx, drop in enumerate(sorted_drops, start=1):
        cdf_rows.append({"drop": drop, "cdf": idx / len(sorted_drops)})
    write_csv_dicts(output_dir / "drop_cdf.csv", cdf_rows, ["drop", "cdf"])
    write_cdf_svg(cdf_rows, output_dir / "drop_cdf.svg")

    risk_ranked = sorted(videos, key=lambda video: risk[video], reverse=True)
    selected_risk = set(risk_ranked[:k])
    all_s = np.asarray([s[video] for video in videos], dtype=float)
    all_s2 = np.asarray([s2[video] for video in videos], dtype=float)
    risk_fallback = np.asarray(
        [s[video] if video in selected_risk else s2[video] for video in videos],
        dtype=float,
    )

    rng = np.random.default_rng(args.seed)
    random_means = []
    for _ in range(args.random_trials):
        selected = set(rng.choice(videos, size=k, replace=False))
        values = [s[video] if video in selected else s2[video] for video in videos]
        random_means.append(float(np.mean(values)))

    mean_s = float(np.mean(all_s))
    mean_s2 = float(np.mean(all_s2))
    mean_risk = float(np.mean(risk_fallback))
    gap = mean_s - mean_s2
    method_rows = [
        {"method": "All S/2", "mean_jf": mean_s2, "rerun_fraction": 0.0, "recovered_gap": 0.0},
        {"method": "All S", "mean_jf": mean_s, "rerun_fraction": 1.0, "recovered_gap": 1.0},
        {
            "method": f"Random fallback {args.budget_fraction:.0%}",
            "mean_jf": float(np.mean(random_means)),
            "rerun_fraction": k / len(videos),
            "recovered_gap": (float(np.mean(random_means)) - mean_s2) / gap if gap else float("nan"),
        },
        {
            "method": f"Risk-gated fallback {args.budget_fraction:.0%}",
            "mean_jf": mean_risk,
            "rerun_fraction": k / len(videos),
            "recovered_gap": (mean_risk - mean_s2) / gap if gap else float("nan"),
        },
    ]
    write_csv_dicts(output_dir / "fallback_methods.csv", method_rows, ["method", "mean_jf", "rerun_fraction", "recovered_gap"])

    drops = [row["drop"] for row in rows]
    summary = {
        "num_videos": len(videos),
        "fallback_budget_videos": k,
        "mean_drop": float(np.mean(drops)),
        "median_drop": float(np.median(drops)),
        "top20_worst_video_drop": top_fraction_mean(drops, 0.2),
        "spearman_risk_vs_drop": spearman_corr([risk[v] for v in videos], [s[v] - s2[v] for v in videos]),
        "all_s_jf": mean_s,
        "all_s2_jf": mean_s2,
        "random_fallback_mean_jf": float(np.mean(random_means)),
        "random_fallback_std_jf": float(np.std(random_means)),
        "risk_fallback_mean_jf": mean_risk,
        "risk_selected_videos": list(risk_ranked[:k]),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
