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


singularity exec --nv --fakeroot \
  -B /export/home:/export/home \
  -B /export/projects:/export/projects \
  -B /export/home/acs/prof/$USER/tmp:/tmp \
  /export/projects/ml/roll_custom.sif \
  python examples/start_agentic_pipeline.py \
    --config_path medical_grounding \
    --config_name medical_grounding