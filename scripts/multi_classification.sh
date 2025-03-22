#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J multi_classification
#SBATCH -o multi_classification.log
#SBATCH -e multi_classification.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(resnet18 resnet50 resnet152)
pretrained_datasets=(imagenet)
transfer_datasets=(celeb_a)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        python -u multi_classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]}
    done
done
