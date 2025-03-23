"""
This script evaluates the robustness improvement achieved by CT across various datasets and attacks.
"""
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch import nn as nn
from torchvision import transforms as transforms
from utils.robustbench import benchmark
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta
from utils.curvature_tuning import replace_module, CT
from loguru import logger
import copy
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def test_robustness(dataset, threat, beta_vals, coeff, seed, model_name, base_batch_size=1000):
    """
    Test the model's robustness with different beta values of CT on the same dataset.
    """
    model = get_pretrained_model(dataset, model_name)
    if dataset == 'imagenet':
        batch_size = base_batch_size // 4
    else:
        batch_size = base_batch_size
    replace_and_test_robustness(model, threat, beta_vals, dataset, coeff=coeff, seed=seed, batch_size=batch_size, model_name=model_name)


def replace_and_test_robustness(model, threat, beta_vals, dataset, coeff=0.5, seed=42, batch_size=1000, model_name="resnet18"):
    """
    Replace ReLU with CT and test the model's robustness on RobustBench.
    """
    model.eval()

    threat_to_eps = {
        'Linf': 8 / 255,
        'L2': 0.5,
        'corruptions': None,
    }

    n_examples = 1000

    if threat != 'corruptions':
        dataset_to_transform = {
            'cifar10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
            ]),
            'cifar100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
            ]),
            'imagenet': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }
    else:     # No need to resize and crop for ImageNet-C as it is already 224x224
        dataset_to_transform = {
            'cifar10': transforms.Compose([
                transforms.Lambda(
                    lambda x: transforms.ToTensor()(x) if isinstance(x, Image.Image) else torch.from_numpy(
                        x.astype(np.float32)) / 255.0
                ),   # Avoid wrong dimension order by ToTensor applied to numpy array
                transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
            ]),
            'cifar100': transforms.Compose([
                transforms.Lambda(
                    lambda x: transforms.ToTensor()(x) if isinstance(x, Image.Image) else torch.from_numpy(
                        x.astype(np.float32)) / 255.0
                ),  # Avoid wrong dimension order by ToTensor applied to numpy array
                transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
            ]),
            'imagenet': transforms.Compose([
                transforms.Lambda(lambda img: transforms.Resize(256)(img) if img.size != (224, 224) else img),  # Avoid resizing if already 224x224
                transforms.Lambda(lambda img: transforms.CenterCrop(224)(img) if img.size != (224, 224) else img),  # Avoid cropping if already 224x224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }

    logger.info(f'Running post-replace robustness test for {model_name} on {dataset} with {threat} attack...')

    acc_list = []
    beta_list = []

    os.makedirs('./cache', exist_ok=True)
    state_path_format_str = f"./cache/{model_name}_{dataset}_{threat}_{n_examples}_{coeff}_{seed}_{{beta:.2f}}.json"

    data_dir = './data' if 'imagenet' not in dataset else './data/imagenet'

    # Test the original model
    logger.debug('Using ReLU...')
    state_path = Path(state_path_format_str.format(beta=1))
    _, base_acc = benchmark(
        model, dataset=dataset, threat_model=threat, eps=threat_to_eps[threat], device=device,
        batch_size=batch_size, preprocessing=dataset_to_transform[dataset], n_examples=n_examples,
        aa_state_path=state_path, seed=seed, data_dir=data_dir
    )
    base_acc *= 100
    logger.debug(f'Robust accuracy: {base_acc:.2f}%')
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        state_path = Path(state_path_format_str.format(beta=beta))
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)
        _, test_acc = benchmark(
            new_model, dataset=dataset, threat_model=threat, eps=threat_to_eps[threat], device=device,
            batch_size=batch_size, preprocessing=dataset_to_transform[dataset], n_examples=n_examples,
            aa_state_path=state_path, seed=seed, data_dir=data_dir
        )
        test_acc *= 100
        logger.debug(f'Robust accuracy: {test_acc:.2f}%')
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)

    acc_list.append(base_acc)
    beta_list.append(1)

    logger.info(f'Best robust accuracy for {dataset} with {threat} attack: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_metric_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, f'{threat}_{n_examples}', metric='Robust Accuracy')

def get_args():
    parser = argparse.ArgumentParser(description='Robustness experiments on RobustBench')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--coeff', type=float, default=0.5, help='Coefficient for CT')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cifar10', 'cifar100', 'imagenet'], help='List of datasets')
    parser.add_argument('--threats', type=str, nargs='+',
                        default=['Linf', 'L2', 'corruptions'], help='List of threats')
    parser.add_argument('--base_batch_size', type=int, default=1000, help='Base batch size for robustness tests')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_coeff{args.coeff}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    betas = np.arange(0.5, 1 - 1e-6, 0.01)

    for ds in args.datasets:
        fix_seed(args.seed)  # Fix the seed each time

        for threat in args.threats:
            if result_exists(f'{ds}', robustness_test=threat):
                logger.info(f'Skipping robustness test for {ds} with {threat} as result already exists.')
            else:
                test_robustness(ds, threat, betas, args.coeff, args.seed, args.model, args.base_batch_size)

if __name__ == '__main__':
    main()
