#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-sft-full
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/sft_full_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/sft_full_%j.err
#SBATCH -t 0-06:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Step 2b: Full SFT — all 8B weights updated (no LoRA)
# Memory strategy: bfloat16 + gradient_checkpointing + adamw_8bit optimizer
# Estimated GPU usage: ~50GB / 80GB
# Dependency: slurm_00_data.sh must complete first.
# Submit: sbatch configs/slurm_02b_sft_full.sh
# =============================================================================

echo "=================================================="
echo "Phase 2b: Full SFT Training (all weights)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_LEVEL=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent
mkdir -p logs checkpoints/sft_full

# Install bitsandbytes for 8-bit optimizer if not already installed
pip install bitsandbytes --break-system-packages -q

python scripts/train_sft_full.py \
    --config configs/sft_full_config.yaml

echo "=================================================="
echo "Full SFT complete. Check checkpoints/sft_full/"
echo "End: $(date)"
echo "=================================================="
