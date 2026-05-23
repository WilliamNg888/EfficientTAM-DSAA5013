"""Risk-gated fallback policy across the four ablation pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "outputs/ablation_generalization"

PAIRS = [
    ("efficienttam_s",  "efficienttam_s_2"),
    ("efficienttam_s",  "efficienttam_s_1"),
    ("efficienttam_ti", "efficienttam_ti_2"),
    ("efficienttam_s",  "efficienttam_s_512x512"),
]

N_TRIALS_RANDOM = 1000


def load_jf(ck: str) -> dict[str, float]:
    with open(RES / f"jf_{ck}.json") as f:
        d = json.load(f)
    return {v: r["JF"] for v, r in d.items() if v != "__mean__"}


def load_risk(ck: str) -> dict[str, float]:
    with open(RES / f"risk_{ck}.json") as f:
        d = json.load(f)
    return {v: x["R"] for v, x in d["per_video"].items()}


def fallback_policy(jf_full, jf_eff, replace_set):
    """Mean J&F if we run efficient by default, and full only on `replace_set`."""
    total = 0.0
    for v in jf_eff:
        if v in replace_set and v in jf_full:
            total += jf_full[v]
        else:
            total += jf_eff[v]
    return total / len(jf_eff)


def main():
    out = []
    for full, eff in PAIRS:
        jf_f = load_jf(full)
        jf_e = load_jf(eff)
        R = load_risk(eff)
        videos = sorted(set(jf_f) & set(jf_e) & set(R))
        n = len(videos)
        k = max(1, int(np.ceil(0.2 * n)))

        baseline_eff = np.mean([jf_e[v] for v in videos]) * 100
        baseline_full = np.mean([jf_f[v] for v in videos]) * 100
        gap = baseline_full - baseline_eff

        # risk-gated top-20%
        sorted_by_R = sorted(videos, key=lambda v: -R[v])
        top_set = set(sorted_by_R[:k])
        risk_mean = np.array([
            jf_f[v] if v in top_set else jf_e[v]
            for v in videos
        ]).mean() * 100

        # random 20% (averaged over N trials)
        rng = np.random.RandomState(0)
        random_means = []
        for _ in range(N_TRIALS_RANDOM):
            sel = set(rng.choice(videos, size=k, replace=False))
            m = np.array([
                jf_f[v] if v in sel else jf_e[v]
                for v in videos
            ]).mean() * 100
            random_means.append(m)
        random_mean = float(np.mean(random_means))
        random_std = float(np.std(random_means))

        # oracle 20% (rerun the actual top-20% biggest drops, upper bound)
        sorted_by_drop = sorted(videos, key=lambda v: -(jf_f[v] - jf_e[v]))
        oracle_set = set(sorted_by_drop[:k])
        oracle_mean = np.array([
            jf_f[v] if v in oracle_set else jf_e[v]
            for v in videos
        ]).mean() * 100

        def recovered(m):
            if gap <= 1e-9:
                return float("nan")
            return (m - baseline_eff) / gap * 100

        row = {
            "pair": f"{full}-{eff}",
            "n_videos": n,
            "k_rerun": k,
            "baseline_eff": baseline_eff,
            "baseline_full": baseline_full,
            "gap": gap,
            "random_mean": random_mean,
            "random_std": random_std,
            "random_recovered": recovered(random_mean),
            "risk_mean": risk_mean,
            "risk_recovered": recovered(risk_mean),
            "oracle_mean": oracle_mean,
            "oracle_recovered": recovered(oracle_mean),
            "top_risk_set": list(top_set),
            "oracle_set": list(oracle_set),
            "overlap_risk_vs_oracle": len(top_set & oracle_set) / k * 100,
        }
        out.append(row)

    # Print table
    print(f"\n{'Pair':<32} {'gap':>6} {'eff':>7} {'rand':>7} {'risk':>7} {'oracle':>7} {'rand%':>6} {'risk%':>6} {'oracle%':>7} {'ovl%':>5}")
    print("-" * 110)
    for r in out:
        print(f"{r['pair']:<32} {r['gap']:>6.2f} "
              f"{r['baseline_eff']:>7.2f} "
              f"{r['random_mean']:>7.2f} {r['risk_mean']:>7.2f} {r['oracle_mean']:>7.2f} "
              f"{r['random_recovered']:>6.1f} {r['risk_recovered']:>6.1f} {r['oracle_recovered']:>7.1f} "
              f"{r['overlap_risk_vs_oracle']:>5.0f}")

    with open(RES / "fallback_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RES / 'fallback_analysis.json'}")


if __name__ == "__main__":
    main()
