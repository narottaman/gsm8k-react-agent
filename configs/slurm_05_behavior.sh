#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-behavior
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/behavior_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/behavior_%j.err
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
# Step 5: Agent Behavioral Evaluation
# Runs agent-specific eval on all checkpoints — not just accuracy but:
#   - tool decision quality (when to use tools)
#   - tool selection (code vs calculator)
#   - step efficiency
#   - accuracy by problem difficulty (easy/medium/hard)
#   - error recovery
#
# Dependency: all training phases complete.
# Submit: sbatch configs/slurm_05_behavior.sh
# =============================================================================

echo "=================================================="
echo "Phase 5: Agent Behavioral Evaluation"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_LEVEL=0
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent
mkdir -p logs data/results

BASE_MODEL="Qwen/Qwen3-8B"

echo ""
echo "--- 1/4: Baseline (zero-shot) ---"
python scripts/eval_agent_behavior.py \
    --model $BASE_MODEL \
    --label baseline \
    --max_samples 100 \
    --wandb

echo ""
echo "--- 2/4: Full SFT ---"
python scripts/eval_agent_behavior.py \
    --model checkpoints/sft_full \
    --label full_sft \
    --max_samples 100 \
    --wandb

echo ""
echo "--- 3/4: LoRA GRPO (on full SFT) ---"
python scripts/eval_agent_behavior.py \
    --model checkpoints/rl/final \
    --lora_base $BASE_MODEL \
    --label lora_grpo \
    --max_samples 100 \
    --wandb

echo ""
echo "--- 4/4: Full GRPO (2-GPU) ---"
if [ -d "checkpoints/rl_full/final" ]; then
    python scripts/eval_agent_behavior.py \
        --model checkpoints/rl_full/final \
        --label full_grpo \
        --max_samples 100 \
        --wandb
else
    echo "Full GRPO checkpoint not found — skipping"
fi

echo ""
echo "=================================================="
echo "All behavioral evals complete."
echo "Results in data/results/behavior_*.json"
echo "End: $(date)"
echo "=================================================="
