#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=16384
#SBATCH --time=24:00:00
#SBATCH --output="tasks/qa_feedback/model_outputs/factuality_quark/training_v0.out"
#SBATCH --open-mode=append

module load eth_proxy

python3 tasks/qa_feedback/training/train_factuality.py --config tasks/qa_feedback/training/factuality_quark_config.yml