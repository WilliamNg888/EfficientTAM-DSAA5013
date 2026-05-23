"""Aggregate per-ckpt J&F + risk into reproduction tables, tail-risk and transferability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
RES = REPO / "outputs/ablation_generalization"
FIGS = RES / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CKPTS = [
    "efficienttam_s",
    "efficienttam_s_2",
    "efficienttam_s_1",
    "efficienttam_ti",
    "efficienttam_ti_2",
    "efficienttam_s_512x512",
]

PAPER_DAVIS = {
    "efficienttam_s": 89.2,
    "efficienttam_s_2": 88.6,
    "efficienttam_s_1": 88.7,
    "efficienttam_ti": 88.4,
    "efficienttam_ti_2": 88.1,
    "efficienttam_s_512x512": 87.2,
}

PAIRS = [
    ("efficienttam_s",  "efficienttam_s_2"),
    ("efficienttam_s",  "efficienttam_s_1"),
    ("efficienttam_ti", "efficienttam_ti_2"),
    ("efficienttam_s",  "efficienttam_s_512x512"),
]


def load_jf(ckpt: str) -> dict:
    p = RES / f"jf_{ckpt}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_risk(ckpt: str) -> dict:
    p = RES / f"risk_{ckpt}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def per_video_jf(jf: dict) -> dict[str, float]:
    return {v: r["JF"] for v, r in jf.items() if v != "__mean__"}


def main():
    summary = {"per_ckpt": {}, "pairs": {}}
    md_lines = ["# EfficientTAM Reproduction + Ablation Analysis", ""]

    # 1) Per-ckpt mean J/F/J&F vs paper
    md_lines += ["## 1. Mean J/F/J&F per ckpt (vs paper Table 1/4/5)\n",
                 "| ckpt | mean J | mean F | mean J&F | paper J&F | diff |",
                 "|---|---|---|---|---|---|"]
    for ck in CKPTS:
        jf = load_jf(ck)
        if not jf or "__mean__" not in jf:
            md_lines.append(f"| {ck} | – | – | – | {PAPER_DAVIS.get(ck,'?')} | NOT RUN |")
            continue
        m = jf["__mean__"]
        meanJF = m["JF"] * 100
        paper = PAPER_DAVIS.get(ck, None)
        diff = meanJF - paper if paper else None
        summary["per_ckpt"][ck] = {
            "meanJ": m["J"] * 100,
            "meanF": m["F"] * 100,
            "meanJF": meanJF,
            "paper_JF": paper,
            "diff_vs_paper": diff,
        }
        md_lines.append(
            f"| {ck} | {m['J']*100:.2f} | {m['F']*100:.2f} | {meanJF:.2f} | "
            f"{paper if paper else '–'} | {f'{diff:+.2f}' if diff is not None else '–'} |"
        )
    md_lines.append("")

    # 2) Per-pair tail-risk
    md_lines += ["## 2. Tail-risk per pair (full − efficient)\n",
                 "| pair | mean drop | median | worst 20% mean | max | argmax video |",
                 "|---|---|---|---|---|---|"]

    for full, eff in PAIRS:
        jf_f = per_video_jf(load_jf(full))
        jf_e = per_video_jf(load_jf(eff))
        if not jf_f or not jf_e:
            md_lines.append(f"| {full} − {eff} | – | – | – | – | not ready |")
            continue
        videos = sorted(set(jf_f) & set(jf_e))
        drops = np.array([(jf_f[v] - jf_e[v]) * 100 for v in videos])
        mean_d = float(drops.mean())
        med_d = float(np.median(drops))
        k = max(1, int(np.ceil(0.2 * len(drops))))
        worst_idx = np.argsort(-drops)[:k]
        worst_mean = float(drops[worst_idx].mean())
        max_d = float(drops.max())
        argmax_v = videos[int(np.argmax(drops))]

        summary["pairs"][f"{full}-{eff}"] = {
            "mean_drop": mean_d,
            "median_drop": med_d,
            "worst20_mean_drop": worst_mean,
            "max_drop": max_d,
            "argmax_video": argmax_v,
            "n_videos": len(videos),
            "per_video_drop": {v: float(d) for v, d in zip(videos, drops)},
        }
        md_lines.append(
            f"| {full} − {eff} | {mean_d:.2f} | {med_d:.2f} | {worst_mean:.2f} | {max_d:.2f} | {argmax_v} |"
        )
    md_lines.append("")

    # 3) Risk score transferability
    md_lines += ["## 3. Risk score Spearman ρ (risk computed on efficient mask vs J&F drop)\n",
                 "| pair | n | ρ(R, Δ) | p-value |",
                 "|---|---|---|---|"]
    for full, eff in PAIRS:
        jf_f = per_video_jf(load_jf(full))
        jf_e = per_video_jf(load_jf(eff))
        risk = load_risk(eff)
        if not risk or "per_video" not in risk or not jf_f or not jf_e:
            md_lines.append(f"| {full} − {eff} | – | – | not ready |")
            continue
        videos = sorted(set(jf_f) & set(jf_e) & set(risk["per_video"].keys()))
        if len(videos) < 5:
            md_lines.append(f"| {full} − {eff} | {len(videos)} | (too few) | – |")
            continue
        deltas = np.array([(jf_f[v] - jf_e[v]) * 100 for v in videos])
        rs = np.array([risk["per_video"][v]["R"] for v in videos])
        rho, pv = spearmanr(rs, deltas)
        summary["pairs"][f"{full}-{eff}"]["spearman_rho"] = float(rho)
        summary["pairs"][f"{full}-{eff}"]["spearman_p"] = float(pv)
        md_lines.append(f"| {full} − {eff} | {len(videos)} | {rho:.3f} | {pv:.3g} |")
    md_lines.append("")

    # 4) Top-risk video overlap across pairs
    md_lines += ["## 4. Top-20% high-risk videos per pair (by R, computed on efficient mask)\n"]
    top_sets = {}
    for full, eff in PAIRS:
        risk = load_risk(eff)
        if not risk or "top_risk" not in risk:
            continue
        top_sets[f"{full}-{eff}"] = set(risk["top_risk"])
        md_lines.append(f"- `{full} − {eff}`: {risk['top_risk']}")
    if len(top_sets) >= 2:
        keys = list(top_sets.keys())
        md_lines.append("\n### Jaccard overlap")
        md_lines.append("| pair A | pair B | A∩B / A∪B |")
        md_lines.append("|---|---|---|")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = top_sets[keys[i]], top_sets[keys[j]]
                if not a or not b:
                    continue
                jacc = len(a & b) / max(len(a | b), 1)
                md_lines.append(f"| {keys[i]} | {keys[j]} | {jacc:.2f} |")
    md_lines.append("")

    # 5) FPS
    fps_path = RES / "fps_benchmark.json"
    if fps_path.exists():
        with open(fps_path) as f:
            fps = json.load(f)
        md_lines += ["## 5. FPS benchmark (GH200, bedroom 200 frames)\n",
                     "| ckpt | params (M) | FPS | ms/frame |",
                     "|---|---|---|---|"]
        for ck in CKPTS:
            if ck in fps:
                v = fps[ck]
                md_lines.append(f"| {ck} | {v['n_params']/1e6:.1f} | {v['fps']:.1f} | {v['ms_per_frame']:.2f} |")
        md_lines.append("")
        summary["fps"] = fps

    # === Figures ===

    # Fig A. Reproduction bars: paper J&F vs ours per ckpt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CKPTS))
    w = 0.35
    paper_vals = [PAPER_DAVIS.get(c, 0) for c in CKPTS]
    ours_vals = [summary["per_ckpt"].get(c, {}).get("meanJF", 0) for c in CKPTS]
    ax.bar(x - w / 2, paper_vals, w, label="paper (Table 1/4/5)", color="#888")
    ax.bar(x + w / 2, ours_vals, w, label="ours (DAVIS 480p)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("efficienttam_", "") for c in CKPTS], rotation=20)
    ax.set_ylabel("DAVIS2017 val mean J&F (%)")
    ax.set_ylim(80, max(max(paper_vals), max(ours_vals)) + 3)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (p, o) in enumerate(zip(paper_vals, ours_vals)):
        if p and o:
            ax.text(i + w / 2, o + 0.2, f"{o:.1f}", ha="center", fontsize=7)
            ax.text(i - w / 2, p + 0.2, f"{p:.1f}", ha="center", fontsize=7, color="dimgray")
    plt.tight_layout()
    plt.savefig(FIGS / "repro_bars.png", dpi=150)
    plt.close()

    # Fig B. FPS vs J&F efficiency frontier
    fps_path = RES / "fps_benchmark.json"
    if fps_path.exists():
        with open(fps_path) as f:
            fps = json.load(f)
        plt.figure(figsize=(7, 5))
        markers = {
            "efficienttam_s": ("o", "#1f77b4", "S (1024)"),
            "efficienttam_s_2": ("s", "#1f77b4", "S/2 (1024, avg-pool)"),
            "efficienttam_s_1": ("D", "#1f77b4", "S/1 (1024, learnable)"),
            "efficienttam_ti": ("o", "#ff7f0e", "Ti (1024)"),
            "efficienttam_ti_2": ("s", "#ff7f0e", "Ti/2 (1024, avg-pool)"),
            "efficienttam_s_512x512": ("^", "#2ca02c", "S (512)"),
        }
        for ck in CKPTS:
            if ck not in fps or ck not in summary["per_ckpt"]:
                continue
            x = fps[ck]["fps"]
            y = summary["per_ckpt"][ck]["meanJF"]
            m, c, lbl = markers.get(ck, ("o", "k", ck))
            plt.scatter(x, y, marker=m, c=c, s=110, edgecolor="k", label=lbl)
            plt.annotate(f"{fps[ck]['n_params']/1e6:.0f}M", (x, y),
                         textcoords="offset points", xytext=(7, -10), fontsize=7)
        plt.xscale("log")
        plt.xlabel("FPS on GH200 (log scale)")
        plt.ylabel("DAVIS2017 val mean J&F (%)")
        plt.grid(alpha=0.3, which="both")
        plt.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        plt.savefig(FIGS / "fps_vs_jf.png", dpi=150)
        plt.close()

    # Fig C. CDF of drops across 4 pairs
    plt.figure(figsize=(7, 4.5))
    colors_p = ["#1f77b4", "#1f77b4", "#ff7f0e", "#2ca02c"]
    styles_p = ["-", "--", "-", "-"]
    for (full, eff), col, sty in zip(PAIRS, colors_p, styles_p):
        key = f"{full}-{eff}"
        if key not in summary["pairs"] or "per_video_drop" not in summary["pairs"][key]:
            continue
        drops = sorted(summary["pairs"][key]["per_video_drop"].values())
        ys = np.arange(1, len(drops) + 1) / len(drops)
        lbl = f"{full.replace('efficienttam_','')} − {eff.replace('efficienttam_','')}"
        plt.plot(drops, ys, color=col, linestyle=sty, marker=".", label=lbl)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Per-video J&F drop (full − efficient, %)")
    plt.ylabel("CDF")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "cdf_drops.png", dpi=150)
    plt.close()

    # Fig D. 2x2 risk scatter grid
    valid_pairs = []
    for full, eff in PAIRS:
        jf_f = per_video_jf(load_jf(full))
        jf_e = per_video_jf(load_jf(eff))
        risk = load_risk(eff)
        if not (jf_f and jf_e and risk and "per_video" in risk):
            continue
        videos = sorted(set(jf_f) & set(jf_e) & set(risk["per_video"].keys()))
        if len(videos) < 5:
            continue
        valid_pairs.append((full, eff, jf_f, jf_e, risk, videos))

    if valid_pairs:
        nrow = (len(valid_pairs) + 1) // 2
        fig, axes = plt.subplots(nrow, 2, figsize=(11, 4.2 * nrow))
        axes = np.array(axes).reshape(-1)
        for idx, (full, eff, jf_f, jf_e, risk, videos) in enumerate(valid_pairs):
            ax = axes[idx]
            deltas = np.array([(jf_f[v] - jf_e[v]) * 100 for v in videos])
            rs = np.array([risk["per_video"][v]["R"] for v in videos])
            top_set = set(risk["top_risk"])
            for i, v in enumerate(videos):
                c = "red" if v in top_set else "steelblue"
                ax.scatter(rs[i], deltas[i], c=c, s=35, edgecolor="k", linewidth=0.3)
                if v in top_set or deltas[i] > max(2.0, np.percentile(deltas, 80)):
                    ax.annotate(v, (rs[i], deltas[i]),
                                textcoords="offset points", xytext=(4, 2), fontsize=7)
            rho = summary["pairs"].get(f"{full}-{eff}", {}).get("spearman_rho")
            ax.set_xlabel("Risk score R(v)")
            ax.set_ylabel(f"J&F drop ({full.replace('efficienttam_','')} − {eff.replace('efficienttam_','')}, %)")
            title = f"{eff.replace('efficienttam_','')}"
            if rho is not None:
                title += f"  (ρ={rho:.3f})"
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.5)
        for j in range(len(valid_pairs), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(FIGS / "risk_scatter_grid.png", dpi=150)
        plt.close()

    # Fig E. Per-video J&F heatmap (videos × ckpts)
    jf_by_ck = {ck: per_video_jf(load_jf(ck)) for ck in CKPTS}
    common_videos = set.intersection(*[set(d.keys()) for d in jf_by_ck.values() if d]) if any(jf_by_ck.values()) else set()
    if common_videos:
        # Sort videos by mean J&F across ckpts (hardest at top)
        videos_sorted = sorted(common_videos,
                               key=lambda v: -np.mean([jf_by_ck[ck][v] for ck in CKPTS if v in jf_by_ck[ck]]))
        mat = np.array([[jf_by_ck[ck].get(v, np.nan) * 100 for ck in CKPTS] for v in videos_sorted])
        fig, ax = plt.subplots(figsize=(7, max(6, 0.22 * len(videos_sorted))))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=60, vmax=100)
        ax.set_xticks(range(len(CKPTS)))
        ax.set_xticklabels([c.replace("efficienttam_", "") for c in CKPTS], rotation=30, ha="right")
        ax.set_yticks(range(len(videos_sorted)))
        ax.set_yticklabels(videos_sorted, fontsize=7)
        for i in range(len(videos_sorted)):
            for j in range(len(CKPTS)):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=6, color="k")
        plt.colorbar(im, ax=ax, label="J&F (%)")
        plt.tight_layout()
        plt.savefig(FIGS / "jf_heatmap.png", dpi=150)
        plt.close()

    # CSV dumps
    if common_videos:
        videos_sorted = sorted(common_videos)
        with open(RES / "per_video_jf.csv", "w") as f:
            f.write("video," + ",".join(CKPTS) + "\n")
            for v in videos_sorted:
                f.write(v + "," + ",".join(f"{jf_by_ck[ck].get(v, float('nan'))*100:.3f}" for ck in CKPTS) + "\n")

        with open(RES / "per_video_drops.csv", "w") as f:
            f.write("video," + ",".join(f"{full}-{eff}" for full, eff in PAIRS) + "\n")
            for v in videos_sorted:
                row = [v]
                for full, eff in PAIRS:
                    if v in jf_by_ck.get(full, {}) and v in jf_by_ck.get(eff, {}):
                        row.append(f"{(jf_by_ck[full][v] - jf_by_ck[eff][v]) * 100:.3f}")
                    else:
                        row.append("nan")
                f.write(",".join(row) + "\n")

    # Write outputs
    with open(RES / "analyze_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    (RES / "analyze_summary.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
