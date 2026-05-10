#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-react-agent
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/train_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/train_%j.err
#SBATCH -t 0-12:00:00
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
# ARIA Repo 1 — GSM8K ReAct Agent with GRPO
#
# Runs the full training loop:
#   1. Load GSM8K dataset (HuggingFace, cached in scratch)
#   2. Rollout: Qwen3-8B via vLLM generates agent trajectories
#   3. Reward: answer_correct + tool_efficiency + format_valid
#   4. GRPO update on trajectory batch
#   5. Log to W&B, save checkpoints
#
# Submit:  sbatch configs/slurm.sh
# Watch:   tail -f logs/train_JOBID.out
# Cancel:  scancel JOBID
# =============================================================================

echo "=================================================="
echo "GSM8K ReAct Agent — GRPO Training"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=================================================="

source ~/envs/sft_lora_rl/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent
mkdir -p logs data/trajectories checkpoints

echo "Python:   $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

python scripts/train.py \
    --config configs/grpo_config.yaml

echo ""
echo "=================================================="
echo "Training complete."
echo "End: $(date)"
echo "=================================================="