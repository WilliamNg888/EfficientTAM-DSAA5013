# Commands for Memory Tail-Risk Experiments

Run all commands from the repository root:

```bash
cd /hpc2hdd/home/ywu706/EfficientTAM-main
```

## Environment Setup

Create and activate a conda environment:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n efficienttam-tailrisk python=3.12 -y
conda activate efficienttam-tailrisk
```

Install PyTorch. For CUDA 12.1 machines, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For a CPU-only smoke test:

```bash
pip install torch torchvision torchaudio
```

Install EfficientTAM and the extra packages used by the analysis scripts:

```bash
pip install -e .
pip install scipy tqdm pillow numpy
```

If CUDA extension compilation causes local installation issues, skip it:

```bash
Efficient_Track_Anything_BUILD_CUDA=0 pip install -e .
pip install scipy tqdm pillow numpy
```


Set paths once:

```bash
DAVIS=/hpc2hdd/home/ywu706/datasets/DAVIS
SPLIT=$DAVIS/ImageSets/2017/val.txt
OUT=outputs/memory_tailrisk
```

Download checkpoints if needed:

```bash
cd checkpoints
./download_checkpoints.sh
cd ..
```



## Experiment 1: S vs S/2 Controlled Comparison

Run EfficientTAM-S:

```bash
python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $DAVIS \
  --split-file $SPLIT \
  --model-preset s \
  --output-dir $OUT/preds_s
```

Run EfficientTAM-S/2:

```bash
python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $DAVIS \
  --split-file $SPLIT \
  --model-preset s2 \
  --output-dir $OUT/preds_s2
```

Evaluate EfficientTAM-S:

```bash
python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $DAVIS \
  --split-file $SPLIT \
  --pred-dir $OUT/preds_s \
  --output-dir $OUT/eval_s
```

Evaluate EfficientTAM-S/2:

```bash
python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $DAVIS \
  --split-file $SPLIT \
  --pred-dir $OUT/preds_s2 \
  --output-dir $OUT/eval_s2
```

Check average J&F:

```bash
cat $OUT/eval_s/summary.json
cat $OUT/eval_s2/summary.json
```

## Experiment 2: Predict Failure Videos from S/2 Outputs

Compute mask-only risk scores from S/2 predictions:

```bash
python3 experiments/memory_tailrisk/risk_score.py \
  --pred-dir $OUT/preds_s2 \
  --output-dir $OUT/risk_s2
```

Compute `Delta_v = J&F_S(v) - J&F_S/2(v)` and Spearman correlation:

```bash
python3 experiments/memory_tailrisk/analyze_fallback.py \
  --s-per-video $OUT/eval_s/per_video.csv \
  --s2-per-video $OUT/eval_s2/per_video.csv \
  --risk-scores $OUT/risk_s2/risk_scores.csv \
  --output-dir $OUT/analysis \
  --budget-fraction 0.2
```

Check the key result:

```bash
cat $OUT/analysis/summary.json
```

The main field for this experiment is:

```text
spearman_risk_vs_drop
```

## Experiment 3: Risk-Gated Fallback

Run fallback analysis with a 20% rerun budget:

```bash
python3 experiments/memory_tailrisk/analyze_fallback.py \
  --s-per-video $OUT/eval_s/per_video.csv \
  --s2-per-video $OUT/eval_s2/per_video.csv \
  --risk-scores $OUT/risk_s2/risk_scores.csv \
  --output-dir $OUT/analysis_fallback20 \
  --budget-fraction 0.2 \
  --random-trials 1000 \
  --seed 0
```

Check method comparison:

```bash
cat $OUT/analysis_fallback20/fallback_methods.csv
cat $OUT/analysis_fallback20/summary.json
```

Important outputs:

```text
$OUT/analysis_fallback20/per_video_delta.csv
$OUT/analysis_fallback20/drop_cdf.csv
$OUT/analysis_fallback20/drop_cdf.svg
$OUT/analysis_fallback20/fallback_methods.csv
$OUT/analysis_fallback20/summary.json
```

## Additional Experiment: SegTrack v2 Average Inference Metrics

SegTrack v2 is an extra dataset outside the EfficientTAM paper's main reported
benchmarks.  For this experiment, keep the same inference setting and report
only mean `J`, `F`, and `J&F` for EfficientTAM-S and EfficientTAM-S/2.

Set paths:

```bash
SEGTRACK_RAW=/path/to/SegTrackv2
SEGTRACK_DAVIS=outputs/segtrack_v2/davis_style
SEGTRACK_OUT=outputs/segtrack_v2
```

Convert SegTrack v2 to DAVIS-style layout:

```bash
python3 experiments/segtrack_v2/prepare_segtrackv2.py \
  --raw-root $SEGTRACK_RAW \
  --output-root $SEGTRACK_DAVIS \
  --overwrite
```

Run EfficientTAM-S:

```bash
python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SEGTRACK_DAVIS/ImageSets/2017/val.txt \
  --model-preset s \
  --output-dir $SEGTRACK_OUT/preds_s
```

Run EfficientTAM-S/2:

```bash
python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SEGTRACK_DAVIS/ImageSets/2017/val.txt \
  --model-preset s2 \
  --output-dir $SEGTRACK_OUT/preds_s2
```

Evaluate EfficientTAM-S:

```bash
python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SEGTRACK_DAVIS/ImageSets/2017/val.txt \
  --pred-dir $SEGTRACK_OUT/preds_s \
  --output-dir $SEGTRACK_OUT/eval_s
```

Evaluate EfficientTAM-S/2:

```bash
python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SEGTRACK_DAVIS/ImageSets/2017/val.txt \
  --pred-dir $SEGTRACK_OUT/preds_s2 \
  --output-dir $SEGTRACK_OUT/eval_s2
```

Check final metrics:

```bash
cat $SEGTRACK_OUT/eval_s/summary.json
cat $SEGTRACK_OUT/eval_s2/summary.json
```
