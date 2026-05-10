#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-compare
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/compare_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/compare_%j.err
#SBATCH -t 0-04:00:00
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
# Step 4: Final Comparison — Baseline vs SFT vs RL
# Loads all checkpoints, evals on same test set, prints comparison table.
# Output: data/results/comparison.json + W&B summary
# Dependency: Steps 1-3 complete.
# Submit: sbatch configs/slurm_04_compare.sh
# =============================================================================

echo "=================================================="
echo "Phase 4: Final Comparison"
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

python scripts/eval_compare.py \
    --config configs/eval_config.yaml

echo "=================================================="
echo "Comparison complete. Check data/results/comparison.json"
echo "End: $(date)"
echo "=================================================="
