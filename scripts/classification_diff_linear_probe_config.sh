#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J classification_diff_linear_probe_config
#SBATCH -o classification_diff_linear_probe_config.log
#SBATCH -e classification_diff_linear_probe_config.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(resnet18)
pretrained_datasets=(mnist cifar10 cifar100 imagenet)
transfer_datasets=(mnist cifar10 cifar100 imagenet)
regs=(0.1 10)
topks=(2 3)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        # Run experiments with different reg values
        for reg in "${regs[@]}"
        do
            python -u classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]} --reg $reg
        done

        # Run experiments with different topk values
        for topk in "${topks[@]}"
        do
            python -u classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]} --topk $topk
        done
    done
done