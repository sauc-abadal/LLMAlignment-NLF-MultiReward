#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11264m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=48128
#SBATCH --output="tasks/qa_feedback/model_outputs/factuality_quark/training_v01.out"
#SBATCH --open-mode=append

module load eth_proxy

python3 tasks/qa_feedback/training/train_factuality_quark.py --config tasks/qa_feedback/training/factuality_quark_config.yml