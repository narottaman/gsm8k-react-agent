#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-sft
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/sft_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/sft_%j.err
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
# Step 2: Supervised Fine-Tuning on GSM8K
# Teaches the model the agent JSON format + math reasoning.
# Output: checkpoints/sft/
# Dependency: slurm_00_data.sh must complete first.
# Submit: sbatch configs/slurm_02_sft.sh
# =============================================================================

echo "=================================================="
echo "Phase 2: SFT Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_LEVEL=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent

python scripts/train_sft.py \
    --config configs/sft_config.yaml

echo "=================================================="
echo "SFT complete. Check checkpoints/sft/"
echo "End: $(date)"
echo "=================================================="
