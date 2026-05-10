#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=install-vllm
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/install_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/install_%j.err
#SBATCH -t 0-00:30:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

echo "Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module load cuda/12.9

source ~/envs/gsm8k_agent/bin/activate

echo "torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Install pre-built vllm wheel — no compilation needed
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121 --no-build-isolation

echo "Verifying..."
python -c "from vllm import LLM; print('vllm: OK')"
echo "Done: $(date)"
