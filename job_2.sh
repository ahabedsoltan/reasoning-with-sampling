#!/bin/bash
#SBATCH -J power_samp_1
#SBATCH -o power_samp_1.%j.log
#SBATCH -e power_samp_1.%j.err
#SBATCH --mail-user=aabedsoltan@ucsd.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bbjr-delta-gpu
#SBATCH --mem=40G
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue

module --ignore_cache load "anaconda3"
source activate verl-grpo

# Print info for debugging
which python
echo "Running batch_idx=1"
echo "Start time: $(date)"

# Change to correct directory
cd /u/abedsol1/research/reasoning-with-sampling/llm_experiments

# Run the script
python power_samp_math.py --batch_idx=1 --mcmc_steps=10 --temp=0.25 --seed=100 --model=qwen_math

echo "End time: $(date)"
