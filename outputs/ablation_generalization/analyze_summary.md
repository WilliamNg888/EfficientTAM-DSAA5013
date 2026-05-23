# EfficientTAM Reproduction + Ablation Analysis

## 1. Mean J/F/J&F per ckpt (vs paper Table 1/4/5)

| ckpt | mean J | mean F | mean J&F | paper J&F | diff |
|---|---|---|---|---|---|
| efficienttam_s | 88.93 | 94.51 | 91.72 | 89.2 | +2.52 |
| efficienttam_s_2 | 88.39 | 93.82 | 91.10 | 88.6 | +2.50 |
| efficienttam_s_1 | 88.24 | 93.78 | 91.01 | 88.7 | +2.31 |
| efficienttam_ti | 87.99 | 93.57 | 90.78 | 88.4 | +2.38 |
| efficienttam_ti_2 | 87.76 | 93.24 | 90.50 | 88.1 | +2.40 |
| efficienttam_s_512x512 | 86.04 | 91.85 | 88.95 | 87.2 | +1.75 |

## 2. Tail-risk per pair (full − efficient)

| pair | mean drop | median | worst 20% mean | max | argmax video |
|---|---|---|---|---|---|
| efficienttam_s − efficienttam_s_2 | 0.62 | 0.14 | 2.85 | 7.60 | kite-surf |
| efficienttam_s − efficienttam_s_1 | 0.71 | 0.05 | 3.60 | 9.63 | kite-surf |
| efficienttam_ti − efficienttam_ti_2 | 0.28 | -0.05 | 2.78 | 9.76 | lab-coat |
| efficienttam_s − efficienttam_s_512x512 | 2.77 | 1.45 | 8.04 | 15.01 | lab-coat |

## 3. Risk score Spearman ρ (risk computed on efficient mask vs J&F drop)

| pair | n | ρ(R, Δ) | p-value |
|---|---|---|---|
| efficienttam_s − efficienttam_s_2 | 30 | 0.268 | 0.152 |
| efficienttam_s − efficienttam_s_1 | 30 | 0.201 | 0.287 |
| efficienttam_ti − efficienttam_ti_2 | 30 | 0.088 | 0.644 |
| efficienttam_s − efficienttam_s_512x512 | 30 | 0.416 | 0.0221 |

## 4. Top-20% high-risk videos per pair (by R, computed on efficient mask)

- `efficienttam_s − efficienttam_s_2`: ['india', 'kite-surf', 'paragliding-launch', 'bmx-trees', 'shooting', 'lab-coat']
- `efficienttam_s − efficienttam_s_1`: ['india', 'paragliding-launch', 'kite-surf', 'bmx-trees', 'motocross-jump', 'shooting']
- `efficienttam_ti − efficienttam_ti_2`: ['india', 'kite-surf', 'motocross-jump', 'bmx-trees', 'paragliding-launch', 'bike-packing']
- `efficienttam_s − efficienttam_s_512x512`: ['india', 'kite-surf', 'paragliding-launch', 'bmx-trees', 'bike-packing', 'motocross-jump']

### Jaccard overlap
| pair A | pair B | A∩B / A∪B |
|---|---|---|
| efficienttam_s-efficienttam_s_2 | efficienttam_s-efficienttam_s_1 | 0.71 |
| efficienttam_s-efficienttam_s_2 | efficienttam_ti-efficienttam_ti_2 | 0.50 |
| efficienttam_s-efficienttam_s_2 | efficienttam_s-efficienttam_s_512x512 | 0.50 |
| efficienttam_s-efficienttam_s_1 | efficienttam_ti-efficienttam_ti_2 | 0.71 |
| efficienttam_s-efficienttam_s_1 | efficienttam_s-efficienttam_s_512x512 | 0.71 |
| efficienttam_ti-efficienttam_ti_2 | efficienttam_s-efficienttam_s_512x512 | 1.00 |

## 5. FPS benchmark (GH200, bedroom 200 frames)

| ckpt | params (M) | FPS | ms/frame |
|---|---|---|---|
| efficienttam_s | 34.1 | 27.5 | 36.31 |
| efficienttam_s_2 | 34.1 | 34.7 | 28.85 |
| efficienttam_s_1 | 34.1 | 34.3 | 29.20 |
| efficienttam_ti | 17.9 | 35.9 | 27.85 |
| efficienttam_ti_2 | 17.9 | 32.4 | 30.91 |
| efficienttam_s_512x512 | 34.1 | 34.7 | 28.83 |
