#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-grpo-full
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/grpo_full_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/grpo_full_%j.err
#SBATCH -t 0-12:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Step 3b: Full-weight GRPO on 2x A100 (160GB total VRAM)
#
# GPU layout:
#   GPU 0: vLLM rollout (16GB) — freed before each update
#   GPU 0+1: Policy full weights (16GB) + optimizer (32GB) spread across both
#
# Dependency: slurm_02b_sft_full.sh must complete first.
# Submit: sbatch configs/slurm_03b_grpo_full.sh
# =============================================================================

echo "=================================================="
echo "Phase 3b: Full-weight GRPO (2x A100)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_LEVEL=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent
mkdir -p logs checkpoints/rl_full data/trajectories

echo "GPU memory before start:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

python scripts/train_grpo_full.py \
    --config configs/grpo_full_config.yaml

echo ""
echo "GPU memory after training:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "=================================================="
echo "Full GRPO complete. Check checkpoints/rl_full/"
echo "End: $(date)"
echo "=================================================="
