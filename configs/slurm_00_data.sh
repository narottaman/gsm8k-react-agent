#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=gsm8k-data
#SBATCH --output=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/data_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/gsm8k-react-agent/logs/data_%j.err
#SBATCH -t 0-00:30:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Step 0: Download + cache GSM8K dataset
# Run this FIRST before any training jobs.
# Fast — no GPU needed, ~5 min.
# Submit: sbatch configs/slurm_00_data.sh
# =============================================================================

echo "=================================================="
echo "GSM8K Data Download"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

source ~/envs/gsm8k_agent/bin/activate

export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export PYTHONPATH="/scratch/ngangada/portfolio/gsm8k-react-agent:$PYTHONPATH"

cd /scratch/ngangada/portfolio/gsm8k-react-agent
mkdir -p logs data/gsm8k data/trajectories data/results checkpoints

python scripts/load_gsm8k.py

echo "=================================================="
echo "Data download complete. Check data/gsm8k/"
echo "End: $(date)"
echo "=================================================="
