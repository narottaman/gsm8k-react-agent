#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-benchmarks
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/benchmarks_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/benchmarks_%j.err
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
# Step 5: Full benchmark comparison
# Runs all model versions on: GSM8K + MATH-500 + ARC-Easy + ARC-Challenge
# Produces clean comparison table in data/results/summary_table.txt
#
# Dependency: all training phases complete.
# Submit: sbatch configs/slurm_05_benchmarks.sh
# =============================================================================

echo "=================================================="
echo "Phase 5: Full Benchmark Suite"
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

python scripts/eval_all_benchmarks.py \
    --config configs/eval_config.yaml \
    --max_samples 100 \
    --benchmarks gsm8k math500 arc_easy arc_challenge

echo ""
echo "=================================================="
echo "Results:"
cat data/results/summary_table.txt
echo ""
echo "End: $(date)"
echo "=================================================="
