#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -t 0-36:00          # time (D-HH:MM)
#SBATCH -o output/output.%N.%j.out  # STDOUT
#SBATCH -e output/output.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1        # request 1 GPU

# Setup Conda environment
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate your conda environment
conda activate special

# Navigate to your project directory
cd ~/core-topics-ai-customer-service-llm

python main.py

