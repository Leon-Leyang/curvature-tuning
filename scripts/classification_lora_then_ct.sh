#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J classification_lora_then_ct
#SBATCH -o classification_lora_then_ct.log
#SBATCH -e classification_lora_then_ct.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(resnet18)
pretrained_datasets_more=(imagenet)
transfer_datasets_more=(arabic_characters arabic_digits beans dtd fashion_mnist fgvc_aircraft food101)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        python -u classification_lora_then_ct.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets_more[@]} --transfer_ds ${transfer_datasets_more[@]}
    done
done
