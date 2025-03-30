#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J ablation
#SBATCH -o ablation.log
#SBATCH -e ablation.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

models=(resnet18)
pretrained_datasets=(imagenet)
transfer_datasets=(arabic_characters arabic_digits beans cub200 dtd fashion_mnist fgvc_aircraft flowers102 food101 med_mnist/dermamnist med_mnist/octmnist med_mnist/pathmnist)
pretrained_datasets_multi=(imagenet)
transfer_datasets_multi=(celeb_a)
robustness_datasets=(cifar10 cifar100 imagenet)
threats=(Linf L2 corruptions)

for model in "${models[@]}"
do
  for coeff in 0 1
  do
      for seed in 42 43 44
      do
          python -u classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets[@]} --transfer_ds ${transfer_datasets[@]} --coeff $coeff
          python -u multi_classification.py --model $model --seed $seed --pretrained_ds ${pretrained_datasets_multi[@]} --transfer_ds ${transfer_datasets_multi[@]} --coeff $coeff
          python -u robustness.py --model $model --seed $seed --datasets ${robustness_datasets[@]} --threats ${threats[@]} --coeff $coeff
      done
  done
done
