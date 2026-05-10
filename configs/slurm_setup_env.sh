#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=setup-env
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/setup_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/setup_%j.err
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

echo "=================================================="
echo "Environment Setup"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

# Show what torch+CUDA we have
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

CUDA_VER=$(python -c "import torch; print(torch.version.cuda.replace('.','')[:3])")
echo "Detected CUDA version string: $CUDA_VER"

# Install numpy first (vllm build needs it)
pip install numpy --upgrade -q

# Install vllm via prebuilt wheel — no source build, no CUDA_HOME needed
# Use the version matching your torch CUDA
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER} -q

echo ""
echo "=================================================="
echo "Verifying..."
python -c "
import torch
print('torch:         ', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:           ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"

python -c "import vllm; print('vllm:          ', vllm.__version__)" || echo "vllm install failed"
python -c "import transformers; print('transformers:  ', transformers.__version__)"
python -c "import datasets; print('datasets:      OK')"
python -c "import wandb; print('wandb:         OK')"
echo "=================================================="
echo "End: $(date)"
echo "=================================================="