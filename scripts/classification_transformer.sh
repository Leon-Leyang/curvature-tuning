#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J classification_transformer
#SBATCH -o classification_transformer.log
#SBATCH -e classification_transformer.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(swin_t swin_s)
pretrained_datasets=(imagenette)
transfer_datasets=(arabic_characters arabic_digits beans cub200 dtd fashion_mnist fgvc_aircraft flowers102 food101)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        python -u classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]}
    done
done
