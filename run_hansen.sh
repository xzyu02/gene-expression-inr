#!/bin/bash
#SBATCH --job-name=hansen_train
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --partition=gpu-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd /oscar/data/tserre/xyu110/gene-expression-inr
mkdir -p slurm_logs

module load miniforge3/25.3.0
source ~/.bashrc
conda activate inr

export CUDA_VISIBLE_DEVICES=0

# Run the script
python train_gene_net_hansen.py --config ./configs/config_hansen.yaml
