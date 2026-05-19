# SegTrack v2 Inference Experiment

This experiment adds an external video object segmentation dataset that is not
part of the EfficientTAM paper's main reported benchmarks.  We keep the same
inference setting used for DAVIS in this repository:

- first-frame ground-truth mask prompt
- forward propagation through the video
- EfficientTAM-S and EfficientTAM-S/2 only
- report mean `J`, `F`, and `J&F`

SegTrack v2 contains 14 videos, 24 objects, and about 1,000 annotated frames.
The public dataset uses `JPEGImages/` and `GroundTruth/` folders, so we first
convert it into a DAVIS-style layout and then reuse the existing VOS scripts.

## 1. Download and Extract SegTrack v2

Official dataset page:

```text
https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html
```

After extraction, the raw root should look like:

```text
SegTrackv2/
  JPEGImages/
  GroundTruth/
```

Some mirrors wrap the zip inside another folder.  In that case, set
`SEGTRACK_RAW` to the folder that directly contains `JPEGImages` and
`GroundTruth`.

## 2. Convert to DAVIS-style Layout

Run from the EfficientTAM repository root:

```bash
SEGTRACK_RAW=/path/to/SegTrackv2
SEGTRACK_DAVIS=outputs/segtrack_v2/davis_style

python3 experiments/segtrack_v2/prepare_segtrackv2.py \
  --raw-root $SEGTRACK_RAW \
  --output-root $SEGTRACK_DAVIS \
  --overwrite
```

The converted output will be:

```text
$SEGTRACK_DAVIS/JPEGImages/480p/<video>/00000.jpg
$SEGTRACK_DAVIS/Annotations/480p/<video>/00000.png
$SEGTRACK_DAVIS/ImageSets/2017/val.txt
```

Check conversion:

```bash
cat $SEGTRACK_DAVIS/conversion_manifest.json
head $SEGTRACK_DAVIS/conversion_manifest.csv
```

## 3. Run EfficientTAM-S and EfficientTAM-S/2

```bash
OUT=outputs/segtrack_v2
SPLIT=$SEGTRACK_DAVIS/ImageSets/2017/val.txt

python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SPLIT \
  --model-preset s \
  --output-dir $OUT/preds_s

python3 experiments/memory_tailrisk/run_vos.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SPLIT \
  --model-preset s2 \
  --output-dir $OUT/preds_s2
```

## 4. Evaluate J/F/J&F

```bash
python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SPLIT \
  --pred-dir $OUT/preds_s \
  --output-dir $OUT/eval_s

python3 experiments/memory_tailrisk/evaluate_jf.py \
  --dataset-root $SEGTRACK_DAVIS \
  --split-file $SPLIT \
  --pred-dir $OUT/preds_s2 \
  --output-dir $OUT/eval_s2
```

Report these two summary files:

```bash
cat $OUT/eval_s/summary.json
cat $OUT/eval_s2/summary.json
```

Or run the result checker, which validates file counts and prints a report-ready
Markdown table:

```bash
python3 experiments/segtrack_v2/check_results.py \
  --converted-root $SEGTRACK_DAVIS \
  --output-root $OUT
```

For the paper/report, this should be presented as an additional inference-only
benchmark table with columns `Dataset`, `Model`, `J`, `F`, and `J&F`.
