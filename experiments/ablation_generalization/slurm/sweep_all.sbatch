#!/bin/bash
# Run inference for 6 ckpts on DAVIS2017 val (4+2 batched on 4 GH200 GPUs).
# Predicted masks live outside the repo (large); set PREDS_ROOT before launch.
#SBATCH --job-name=efftam_sweep
#SBATCH --account=a0093
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/sweep_%j.log

set -eo pipefail

source /capstor/scratch/cscs/hcwang942/miniforge3/etc/profile.d/conda.sh
conda activate formalrx311

REPO=$(git rev-parse --show-toplevel)
cd $REPO

: ${DAVIS:?must set DAVIS=/path/to/DAVIS}
: ${PREDS_ROOT:?must set PREDS_ROOT=/path/for/preds_full}
mkdir -p $PREDS_ROOT slurm_logs

EXP=experiments/ablation_generalization

run_one () {
    local name=$1 cfg=$2 gpu=$3
    echo "[gpu=$gpu] $name"
    CUDA_VISIBLE_DEVICES=$gpu python $EXP/run_inference.py \
        --ckpt $REPO/checkpoints/$name.pt \
        --cfg configs/efficienttam/$cfg \
        --davis $DAVIS \
        --out $PREDS_ROOT/$name \
        > slurm_logs/sweep_${name}.log 2>&1 &
}

run_one efficienttam_s         efficienttam_s.yaml         0
run_one efficienttam_s_2       efficienttam_s_2.yaml       1
run_one efficienttam_s_1       efficienttam_s_1.yaml       2
run_one efficienttam_ti        efficienttam_ti.yaml        3
wait

run_one efficienttam_ti_2      efficienttam_ti_2.yaml      0
run_one efficienttam_s_512x512 efficienttam_s_512x512.yaml 1
wait

ls $PREDS_ROOT/
