# EfficientTAM Memory Tail-Risk Experiments

This folder contains an inference-only reproduction extension for the MLRC story:
EfficientTAM-S/2 is close to EfficientTAM-S on average, but has predictable
per-video tail failures.  All scripts live outside the original project code.

## 1. Run Controlled S vs S/2 Inference

Expected DAVIS layout:

```text
DAVIS/
  JPEGImages/480p/<video>/*.jpg
  Annotations/480p/<video>/*.png
  ImageSets/2017/val.txt
```

Run both models:

```bash
python experiments/memory_tailrisk/run_vos.py \
  --dataset-root /path/to/DAVIS \
  --split-file /path/to/DAVIS/ImageSets/2017/val.txt \
  --model-preset s \
  --output-dir outputs/memory_tailrisk/preds_s

python experiments/memory_tailrisk/run_vos.py \
  --dataset-root /path/to/DAVIS \
  --split-file /path/to/DAVIS/ImageSets/2017/val.txt \
  --model-preset s2 \
  --output-dir outputs/memory_tailrisk/preds_s2
```

By default the scripts use `checkpoints/efficienttam_s.pt` and
`checkpoints/efficienttam_s_2.pt`.  Download them with the repository's
`checkpoints/download_checkpoints.sh` first.

## 2. Evaluate Per-Video J&F

```bash
python experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root /path/to/DAVIS \
  --split-file /path/to/DAVIS/ImageSets/2017/val.txt \
  --pred-dir outputs/memory_tailrisk/preds_s \
  --output-dir outputs/memory_tailrisk/eval_s

python experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root /path/to/DAVIS \
  --split-file /path/to/DAVIS/ImageSets/2017/val.txt \
  --pred-dir outputs/memory_tailrisk/preds_s2 \
  --output-dir outputs/memory_tailrisk/eval_s2
```

The first frame is skipped by default because it is used as the mask prompt.

## 3. Compute S/2 Mask-Only Risk

```bash
python experiments/memory_tailrisk/risk_score.py \
  --pred-dir outputs/memory_tailrisk/preds_s2 \
  --output-dir outputs/memory_tailrisk/risk_s2
```

The risk score uses only S/2 predictions:

- area jump: `abs(log((|M_t| + eps) / (|M_{t-1}| + eps)))`
- temporal IoU instability: `1 - IoU(M_t, M_{t-1})`
- fragmentation: number of connected components in `M_t`

Each metric is converted to a video-level top-20% mean, robust-z normalized
across videos, and averaged into `risk_score`.

## 4. Tail-Risk and Fallback Analysis

```bash
python experiments/memory_tailrisk/analyze_fallback.py \
  --s-per-video outputs/memory_tailrisk/eval_s/per_video.csv \
  --s2-per-video outputs/memory_tailrisk/eval_s2/per_video.csv \
  --risk-scores outputs/memory_tailrisk/risk_s2/risk_scores.csv \
  --output-dir outputs/memory_tailrisk/analysis \
  --budget-fraction 0.2
```

Important outputs:

- `per_video_delta.csv`: `Delta_v = J&F_S(v) - J&F_S/2(v)`
- `drop_cdf.csv` and `drop_cdf.svg`: per-video drop CDF
- `fallback_methods.csv`: All S/2, All S, Random fallback, Risk-gated fallback
- `summary.json`: mean/median/top-20% drop and Spearman correlation

## Paper-Facing Claims To Check

The result is useful if `summary.json` supports these statements:

1. Mean S/2 drop is small, but top-20% worst-video drop is much larger.
2. Spearman correlation between S/2 risk score and S-vs-S/2 drop is positive.
3. With the same rerun budget, risk-gated fallback recovers more J&F than random fallback.

## Additional Dataset: SegTrack v2

To add a dataset that is not part of the EfficientTAM paper's main reported
benchmarks, use SegTrack v2.  The adapter lives in:

```text
experiments/segtrack_v2/prepare_segtrackv2.py
```

It converts raw SegTrack v2 (`JPEGImages/` + `GroundTruth/`) into the same
DAVIS-style layout expected by `run_vos.py` and `evaluate_jf.py`.  See:

```text
experiments/segtrack_v2/README.md
```

For this additional experiment, only report mean `J`, `F`, and `J&F` for
EfficientTAM-S and EfficientTAM-S/2.  No tail-risk or fallback analysis is
required.
