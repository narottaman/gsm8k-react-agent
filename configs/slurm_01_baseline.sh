#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-baseline
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/baseline_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/baseline_%j.err
#SBATCH -t 0-02:00:00
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
# Step 1: Baseline evaluation — Qwen3-8B zero-shot, no fine-tuning
# Runs ReAct agent on GSM8K test set, saves results + W&B logs.
# Dependency: slurm_00_data.sh must complete first.
# Submit: sbatch configs/slurm_01_baseline.sh
# =============================================================================

echo "=================================================="
echo "Phase 1: Baseline Evaluation (zero-shot)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent

python scripts/eval_baseline.py \
    --config configs/eval_config.yaml

echo "=================================================="
echo "Baseline eval complete. Check data/results/baseline.json"
echo "End: $(date)"
echo "=================================================="
