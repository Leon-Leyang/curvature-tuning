#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J classification_test_unseen
#SBATCH -o classification_test_unseen.log
#SBATCH -e classification_test_unseen.log
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
transfer_datasets=(arabic_characters arabic_digits beans cub200 dtd fashion_mnist fgvc_aircraft food101 med_mnist/dermamnist med_mnist/octmnist med_mnist/pathmnist)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        python -u classification_test_unseen.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]}
    done
done
