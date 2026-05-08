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

cd ~/workspace/HRIA/vlm/ROLL

SIF=/export/projects/ml/roll.sif

export RAY_TMPDIR=~/r

apptainer exec \
  --nv \
  --bind ~/workspace:/workspace \
  --bind ~/r:/tmp \
  --bind ~/.cache/modelscope:/root/.cache/modelscope \
  $SIF \
  bash -c "

    echo '===== CONTAINER ENV ====='

    pip install --force-reinstall triton

    # Force clean container paths
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

    export MODELSCOPE_CACHE=/root/.cache/modelscope
    export HF_HOME=/root/.cache/huggingface

    export TRITON_CACHE_DIR=/tmp/triton_cache
    export TRITON_ALWAYS_COMPILE=1
    rm -rf /tmp/triton_cache

    unset CUDA_HOME
    unset CUDA_PATH
    unset CUDA_ROOT

    # IMPORTANT: remove any possibility of stub discovery
    export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/lib/x86_64-linux-gnu
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

    # prevent CUDA toolkit probing stubs
    export CUDA_MODULE_LOADING=LAZY

    echo '===== PYTHON INFO ====='

    which python || true

    python - <<'PY'
import sys
import os
import ctypes
import ctypes.util

print('sys.executable =', sys.executable)
print('sys.prefix =', sys.prefix)

print('\\n=== CUDA ===')

print('find_library(cuda) =',
      ctypes.util.find_library('cuda'))

lib = ctypes.CDLL('libcuda.so')

print('loaded libcuda =', lib)

print('has cuModuleGetFunction =',
      hasattr(lib, 'cuModuleGetFunction'))

print('\\n=== TORCH ===')

try:
    import torch
    print('torch =', torch.__version__)
    print('torch cuda =', torch.version.cuda)
    print('torch file =', torch.__file__)
    print('cuda available =', torch.cuda.is_available())
except Exception as e:
    print('torch error:', e)

print('\\n=== ENV ===')

for k in [
    'PATH',
    'LD_LIBRARY_PATH',
    'LIBRARY_PATH',
    'CUDA_HOME',
    'CUDA_PATH',
    'CONDA_PREFIX',
]:
    print(f'{k} = {os.environ.get(k)}')
PY

    echo '===== NVIDIA-SMI ====='
    nvidia-smi

    echo '===== LIBCUDA FILES ====='
    find / -name 'libcuda.so*' 2>/dev/null

    echo '===== INSTALL ====='

    cd /workspace/HRIA/vlm/ROLL/mcore_adapter
    pip install -e .

    cd /workspace/HRIA/vlm/ROLL

    export PYTHONPATH=/workspace/HRIA/vlm/ROLL

    pip install -e .

    echo '===== START PIPELINE ====='

    python examples/start_agentic_pipeline.py \
      --config_path medical_grounding \
      --config_name medical_grounding
"