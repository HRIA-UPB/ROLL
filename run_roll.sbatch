#!/bin/bash
#SBATCH --job-name=roll
#SBATCH --account=hria
#SBATCH --partition=h200
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1100G
#SBATCH --dependency=singleton
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00

export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
export LD_PRELOAD=$CONDA_PREFIX/lib/libcudart.so.12

cd ~/workspace/HRIA/vlm/ROLL

# Initialize conda (Miniforge path)
source ~/miniforge3/etc/profile.d/conda.sh

# Activate env
conda activate roll

# Debug (optional but useful)
which python
python -V
echo $CONDA_DEFAULT_ENV

# Run
python examples/start_agentic_pipeline.py \
  --config_path medical_grounding \
  --config_name medical_grounding