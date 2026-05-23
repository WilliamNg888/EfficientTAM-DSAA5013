# Ablation Generalization

Companion to `experiments/memory_tailrisk/`. The original $S$ vs $S/2$ story
(mean-close but heavy-tailed, with a risk-gated fallback) is extended to the
remaining efficiency axes the paper ablates:

- cross-attention variant: $S/1$ (learnable; paper Table 4 eq.6)
- backbone scaling: $\mathrm{Ti}$ and $\mathrm{Ti}/2$
- input resolution: $S$ at $512{\times}512$ (paper Table 5)

Six checkpoints in total are evaluated on DAVIS2017 val (30 videos, 480p).
The risk score, fallback policy, and FPS benchmark are recomputed for each.

## What this folder reproduces

1. Per-checkpoint mean $\mathcal{J}\&\mathcal{F}$ on DAVIS2017 val, compared to
   paper Tables 1 / 4 / 5.
2. Per-video drops for the four ablation pairs and their worst-20% means.
3. Mask-only risk score $R(v)$ computed on the efficient variant, and its
   Spearman correlation with the per-video drop.
4. Risk-gated fallback policy at a 20% rerun budget, and a sweep from 5% to 50%.
5. Per-frame Jaccard trajectories on the four highest-risk videos.
6. Inference FPS on the six checkpoints on a single GH200.

The risk score and fallback policy follow the formulas in
`experiments/memory_tailrisk/`; the only addition here is applying them
beyond the $S$ vs $S/2$ pair.

## Files

```
run_inference.py    propagate a ckpt over DAVIS2017 val, dump per-frame .npz
eval_jf.py          per-video J / F / J&F (DAVIS metric, F dilation = 0.008*diag)
risk_score.py       mask-only R(v): area jump + IoU instability + fragmentation
analyze.py          aggregate the 6 J&F + 6 risk files into tables and figures
fallback.py         risk-gated vs random vs oracle fallback at 20% budget
budget_curve.py     sweep budget 5..50% per pair, save curve
trajectory.py       per-frame J for kite-surf / lab-coat / paragliding-launch / bmx-trees
run_benchmark.py    FPS over the 6 ckpts on the bedroom demo video
slurm/              CSCS GH200 reference job scripts
```

Concrete commands in `command.md`. Pre-computed outputs in
`outputs/ablation_generalization/`.

## Inputs assumed by the scripts

- Checkpoints at `checkpoints/efficienttam_*.pt` (downloaded via the upstream
  `checkpoints/download_checkpoints.sh`). Six files: `s`, `s_2`, `s_1`, `ti`,
  `ti_2`, `s_512x512`.
- DAVIS2017 at any path; passed via `--davis $DAVIS`.
- Predicted masks under `$PREDS_ROOT/<ckpt>/<video>/<frame>.npz`. Mask
  predictions are large (~30 GB for the 6 ckpts) and are not stored in this
  repo; regenerate via `run_inference.py`.

## Environment notes

`torch.compile` and the bundled CUDA extension are not used because Triton
fails to build under the aarch64 conda toolchain on GH200. The scripts pass
`++model.compile_image_encoder=False` to `build_efficienttam_video_predictor`
to avoid the compile path. The remaining inference numbers therefore come
from the pure-PyTorch path, in contrast to the A100 numbers in the paper.
