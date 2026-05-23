"""Sweep rerun budgets 5..50% and plot recovered-gap curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "outputs/ablation_generalization"
FIGS = RES / "figs"

PAIRS = [
    ("efficienttam_s",  "efficienttam_s_2",        "S − S/2",   "#1f77b4", "-"),
    ("efficienttam_s",  "efficienttam_s_1",        "S − S/1",   "#1f77b4", "--"),
    ("efficienttam_ti", "efficienttam_ti_2",       "Ti − Ti/2", "#ff7f0e", "-"),
    ("efficienttam_s",  "efficienttam_s_512x512",  "S − S$_{512}$","#2ca02c", "-"),
]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_TRIALS = 500


def load_jf(ck: str) -> dict[str, float]:
    with open(RES / f"jf_{ck}.json") as f:
        d = json.load(f)
    return {v: r["JF"] for v, r in d.items() if v != "__mean__"}


def load_risk(ck: str) -> dict[str, float]:
    with open(RES / f"risk_{ck}.json") as f:
        d = json.load(f)
    return {v: x["R"] for v, x in d["per_video"].items()}


def main():
    out = []
    plt.figure(figsize=(8, 5))

    for full, eff, name, color, ls in PAIRS:
        jf_f = load_jf(full)
        jf_e = load_jf(eff)
        R = load_risk(eff)
        videos = sorted(set(jf_f) & set(jf_e) & set(R))
        n = len(videos)

        baseline_eff = np.mean([jf_e[v] for v in videos]) * 100
        baseline_full = np.mean([jf_f[v] for v in videos]) * 100
        gap = baseline_full - baseline_eff

        sorted_by_R = sorted(videos, key=lambda v: -R[v])

        random_rec_at_b = []
        risk_rec_at_b = []
        for b in BUDGETS:
            k = max(1, int(np.ceil(b * n)))

            # risk-gated top-k
            top_set = set(sorted_by_R[:k])
            risk_m = np.array([
                jf_f[v] if v in top_set else jf_e[v]
                for v in videos
            ]).mean() * 100
            rec_risk = (risk_m - baseline_eff) / gap * 100 if gap > 1e-9 else 0

            # random k averaged over trials
            rng = np.random.RandomState(0)
            rs = []
            for _ in range(N_TRIALS):
                sel = set(rng.choice(videos, size=k, replace=False))
                m = np.array([
                    jf_f[v] if v in sel else jf_e[v]
                    for v in videos
                ]).mean() * 100
                rs.append(m)
            rand_m = float(np.mean(rs))
            rec_rand = (rand_m - baseline_eff) / gap * 100 if gap > 1e-9 else 0

            risk_rec_at_b.append(rec_risk)
            random_rec_at_b.append(rec_rand)

        out.append({
            "pair": f"{full}-{eff}",
            "label": name,
            "gap": gap,
            "budgets": BUDGETS,
            "risk_recovered": risk_rec_at_b,
            "random_recovered": random_rec_at_b,
        })
        print(f"{name:<18} gap={gap:.2f}")
        print(f"  budgets:   ", " ".join(f"{b*100:>5.0f}%" for b in BUDGETS))
        print(f"  random:    ", " ".join(f"{x:>5.1f} " for x in random_rec_at_b))
        print(f"  risk-gated:", " ".join(f"{x:>5.1f} " for x in risk_rec_at_b))

        # Plot pair: risk-gated solid color, random dotted same color (lighter)
        bps = [b * 100 for b in BUDGETS]
        plt.plot(bps, risk_rec_at_b, color=color, linestyle=ls, marker="o",
                 label=f"{name} (risk)", linewidth=1.8)
        plt.plot(bps, random_rec_at_b, color=color, linestyle=ls, marker="x",
                 alpha=0.45, linewidth=1.0, label=f"{name} (random)")

    plt.xlabel("Rerun budget (% of videos)")
    plt.ylabel("Recovered gap (%)")
    plt.legend(fontsize=7, ncol=2, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "budget_curve.png", dpi=150)
    plt.close()

    with open(RES / "budget_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RES / 'budget_curve.json'}, {FIGS / 'budget_curve.png'}")


if __name__ == "__main__":
    main()
