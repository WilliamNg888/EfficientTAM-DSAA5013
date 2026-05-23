# Commands for Ablation Generalization

All commands are run from the repository root.

## Environment

Same conda env as `experiments/memory_tailrisk`. On CSCS GH200 the gcc-13 +
no-CUDA-ext trick is needed (see `slurm/setup_env.sbatch`); on most other
sites a plain `pip install -e .` works.

```bash
conda activate efficienttam-tailrisk
pip install -e . --no-deps
pip install scipy tqdm pillow numpy matplotlib
```

## Paths

```bash
DAVIS=/path/to/DAVIS                              # DAVIS2017
PREDS_ROOT=/path/to/preds_full                    # large; outside repo
EXP=experiments/ablation_generalization
OUT=outputs/ablation_generalization
mkdir -p $PREDS_ROOT $OUT/figs
```

## Step 1. Inference over the six checkpoints

One ckpt at a time:

```bash
python $EXP/run_inference.py \
  --ckpt checkpoints/efficienttam_s.pt \
  --cfg configs/efficienttam/efficienttam_s.yaml \
  --davis $DAVIS \
  --out $PREDS_ROOT/efficienttam_s
```

Repeat with `efficienttam_s_2`, `efficienttam_s_1`, `efficienttam_ti`,
`efficienttam_ti_2`, `efficienttam_s_512x512`. The four-GPU batched version
is in `slurm/sweep_all.sbatch`.

## Step 2. Per-video J&F

```bash
for ck in efficienttam_s efficienttam_s_2 efficienttam_s_1 \
          efficienttam_ti efficienttam_ti_2 efficienttam_s_512x512; do
    python $EXP/eval_jf.py \
        --pred $PREDS_ROOT/$ck \
        --davis $DAVIS \
        --out $OUT/jf_$ck.json
done
```

## Step 3. Per-video risk score

```bash
for ck in efficienttam_s efficienttam_s_2 efficienttam_s_1 \
          efficienttam_ti efficienttam_ti_2 efficienttam_s_512x512; do
    python $EXP/risk_score.py \
        --pred $PREDS_ROOT/$ck \
        --davis $DAVIS \
        --out $OUT/risk_$ck.json
done
```

## Step 4. FPS benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python $EXP/run_benchmark.py
```

Writes `$OUT/fps_benchmark.json`. The benchmark video is
`notebooks/videos/bedroom` from the upstream repo (200 frames, single object).

## Step 5. Aggregate analysis and figures

```bash
python $EXP/analyze.py        # tables + repro_bars, cdf_drops, jf_heatmap, ...
python $EXP/fallback.py       # risk vs random vs oracle at 20% budget
python $EXP/budget_curve.py   # budget 5..50% sweep
python $EXP/trajectory.py --davis $DAVIS --preds-root $PREDS_ROOT
```

Outputs land in `$OUT/`:

```text
analyze_summary.{json,md}     reproduction tables, pair drops, Spearman ρ
fallback_analysis.json        per-pair recovered gap and overlap with oracle
budget_curve.json             recovered gap vs rerun budget
trajectory.json               per-frame J for the four hardest videos
per_video_{drops,jf}.csv
figs/                         repro_bars, cdf_drops, jf_heatmap,
                              fps_vs_jf, budget_curve, risk_scatter_grid,
                              trajectory_*.png
```

## SLURM (CSCS GH200 reference)

```bash
sbatch $EXP/slurm/setup_env.sbatch
DAVIS=... PREDS_ROOT=... sbatch $EXP/slurm/sweep_all.sbatch
DAVIS=... PREDS_ROOT=... sbatch $EXP/slurm/post_sweep_par.sbatch
```

## Key claims to check after running

- Mean drop on every pair is below 1 J&F point but worst-20% mean exceeds
  2.7 points; see `analyze_summary.md` section 2.
- Top-20% high-risk videos overlap heavily across pairs (Jaccard 0.50 to 1.00);
  same section 4.
- Risk-gated fallback beats random at 20% budget for the pooling and
  resolution pairs (~3x), and is near-zero or negative for Ti/2; see
  `fallback_analysis.json`.
- Spearman $\rho$ between $R$ and the per-video drop is significant for the
  resolution pair (p = 0.022) and not for the others; section 3.
