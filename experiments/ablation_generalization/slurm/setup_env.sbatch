#!/bin/bash
# CSCS GH200 (aarch64) reference. Edit REPO and conda env for other sites.
#SBATCH --job-name=efftam_env
#SBATCH --account=a0093
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/env_%j.log

set -eo pipefail

source /capstor/scratch/cscs/hcwang942/miniforge3/etc/profile.d/conda.sh
conda activate formalrx311

export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export Efficient_Track_Anything_BUILD_CUDA=1
export Efficient_Track_Anything_BUILD_ALLOW_ERRORS=1

REPO=$(git rev-parse --show-toplevel)
cd $REPO

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -e . --no-deps
pip install --no-deps hydra-core iopath omegaconf antlr4-python3-runtime
python -c "import cv2" 2>/dev/null || pip install opencv-python
python -c "import scipy" 2>/dev/null || pip install scipy

python -c "from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor; print('import OK')"
