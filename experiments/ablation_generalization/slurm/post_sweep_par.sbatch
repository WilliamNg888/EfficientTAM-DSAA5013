#!/bin/bash
# Eval J&F, risk score, FPS benchmark and aggregate analysis for the 6 ckpts.
#SBATCH --job-name=efftam_post
#SBATCH --account=a0093
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/post_par_%j.log

set -eo pipefail

source /capstor/scratch/cscs/hcwang942/miniforge3/etc/profile.d/conda.sh
conda activate formalrx311

REPO=$(git rev-parse --show-toplevel)
cd $REPO

: ${DAVIS:?must set DAVIS=/path/to/DAVIS}
: ${PREDS_ROOT:?must set PREDS_ROOT=/path/to/preds_full}

EXP=experiments/ablation_generalization
OUT=outputs/ablation_generalization
mkdir -p $OUT/figs slurm_logs

CKPTS="efficienttam_s efficienttam_s_2 efficienttam_s_1 efficienttam_ti efficienttam_ti_2 efficienttam_s_512x512"

for ck in $CKPTS; do
    [ -d "$PREDS_ROOT/$ck" ] || { echo "[skip] $ck"; continue; }
    (python -u $EXP/eval_jf.py --pred $PREDS_ROOT/$ck --davis $DAVIS \
        --out $OUT/jf_$ck.json > slurm_logs/eval_jf_${ck}.log 2>&1) &
done
wait

for ck in $CKPTS; do
    [ -d "$PREDS_ROOT/$ck" ] || continue
    (python -u $EXP/risk_score.py --pred $PREDS_ROOT/$ck --davis $DAVIS \
        --out $OUT/risk_$ck.json > slurm_logs/risk_${ck}.log 2>&1) &
done
wait

CUDA_VISIBLE_DEVICES=0 python -u $EXP/run_benchmark.py 2>&1 | tee slurm_logs/benchmark.log

python -u $EXP/analyze.py 2>&1 | tee slurm_logs/analyze.log
python -u $EXP/fallback.py 2>&1 | tee slurm_logs/fallback.log
python -u $EXP/budget_curve.py 2>&1 | tee slurm_logs/budget.log
python -u $EXP/trajectory.py --davis $DAVIS --preds-root $PREDS_ROOT 2>&1 | tee slurm_logs/trajectory.log

ls -la $OUT/
