#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-grpo
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/grpo_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/grpo_%j.err
#SBATCH -t 0-03:00:00
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
# Step 3: GRPO RL Training
# Trains agent to decide WHEN to use tools using reward signals.
# Starts from SFT checkpoint if available, else from base model.
# Output: checkpoints/rl/
# Dependency: slurm_02_sft.sh should complete first (not required).
# Submit: sbatch configs/slurm_03_grpo.sh
# =============================================================================

echo "=================================================="
echo "Phase 3: GRPO RL Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export HF_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_LEVEL=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent

python scripts/train_grpo.py \
    --config configs/grpo_config.yaml

echo "=================================================="
echo "GRPO complete. Check checkpoints/rl/"
echo "End: $(date)"
echo "=================================================="