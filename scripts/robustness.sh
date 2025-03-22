#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -J robustness
#SBATCH -o robustness.log
#SBATCH -e robustness.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(resnet18 resnet50 resnet152)
datasets=(cifar10 cifar100 imagenet)
threats=(Linf L2 corruptions)

for model in "${models[@]}"
do
    for seed in 42 43 44
    do
        python -u robustness.py --model $model --seed $seed --datasets ${datasets[@]} --threats ${threats[@]}
    done
done
