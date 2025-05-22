"""
This script evaluates the robustness improvement achieved by CT across various datasets and attacks.
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch import nn as nn
from torchvision import transforms as transforms
from utils.robustbench import benchmark
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger
from utils.curvature_tuning import replace_module, CTU
from loguru import logger
import copy
import argparse
import json

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

THREAT_TO_EPS = {
    'Linf': 8 / 255,
    'L2': 0.5,
    'corruptions': None,
}


def get_transform(threat, dataset):
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
    else:  # No need to resize and crop for ImageNet-C as it is already 224x224
        dataset_to_transform = {
            'cifar10': transforms.Compose([
                transforms.Lambda(
                    lambda x: transforms.ToTensor()(x) if isinstance(x, Image.Image) else torch.from_numpy(
                        x.astype(np.float32)) / 255.0
                ),  # Avoid wrong dimension order by ToTensor applied to numpy array
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
                transforms.Lambda(lambda img: transforms.Resize(256)(img) if img.size != (224, 224) else img),
                # Avoid resizing if already 224x224
                transforms.Lambda(lambda img: transforms.CenterCrop(224)(img) if img.size != (224, 224) else img),
                # Avoid cropping if already 224x224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }
    return dataset_to_transform[dataset]


def get_args():
    parser = argparse.ArgumentParser(description='Robustness experiments on RobustBench')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threat', type=str,
                        default='Linf', help='Threat to test against (Linf, L2, corruptions)')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='Dataset on which to test the model')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for robustness tests')
    parser.add_argument('--n_examples', type=int, default=1000, help='Number of samples to test')

    return parser.parse_args()


def main():
    args = get_args()

    result_path = {'baseline': f'./robust_results/base_{args.threat}_{args.dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
                   'ct': f'./robust_results/ct_{args.threat}_{args.dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json'}

    # Check if all result files exist
    if all(os.path.exists(path) for path in result_path.values()):
        print('All result files already exist. Exiting...')
        return

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_{args.threat}_{args.dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    fix_seed(args.seed)  # Fix the seed each time

    logger.info(f'Running on {device}')

    model = get_pretrained_model(args.dataset, args.model)
    model.eval()

    transform = get_transform(args.threat, args.dataset)

    data_dir = './data/imagenet' if 'imagenet' in args.dataset else './data'

    # Make directory for evaluation cache
    os.makedirs('./cache', exist_ok=True)

    logger.info(f'Testing the baseline')
    state_path = Path(f"./cache/{args.threat}_{args.dataset}_sample{args.n_examples}_{args.model}_base_seed{args.seed}.json")
    _, base_acc = benchmark(
        model, dataset=args.dataset, threat_model=args.threat, eps=THREAT_TO_EPS[args.threat], device=device,
        batch_size=args.batch_size, preprocessing=transform, n_examples=args.n_examples,
        aa_state_path=state_path, seed=args.seed, data_dir=data_dir
    )
    base_acc *= 100
    logger.info(f'Robust accuracy: {base_acc:.2f}%')

    beta_range = np.arange(0.7, 1.0 - 1e-6, 0.01)
    best_acc = -1
    best_beta = None
    acc_list = []

    for beta in beta_range:
        logger.info(f'Testing CT with beta: {beta:.2f}')
        state_path = Path(f"./cache/{args.threat}_{args.dataset}_sample{args.n_examples}_{args.model}_beta{beta:.2f}_seed{args.seed}.json")
        shared_raw_beta = nn.Parameter(torch.logit(torch.tensor(beta)), requires_grad=False)
        shared_raw_coeff = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        ct_model = replace_module(copy.deepcopy(model), old_module=nn.ReLU, new_module=CTU,
                                  shared_raw_beta=shared_raw_beta, shared_raw_coeff=shared_raw_coeff).to(device)
        _, test_acc = benchmark(
            ct_model, dataset=args.dataset, threat_model=args.threat, eps=THREAT_TO_EPS[args.threat], device=device,
            batch_size=args.batch_size, preprocessing=transform, n_examples=args.n_examples,
            aa_state_path=state_path, seed=args.seed, data_dir=data_dir
        )
        test_acc *= 100
        acc_list.append(test_acc)
        logger.info(f'Robust accuracy: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta

    # Log the summary
    logger.info(f'Baseline accuracy: {base_acc:.2f}%')
    logger.info(f'Best accuracy for CT: {best_acc:.2f}% with beta: {best_beta:.2f}')

    # Save the results
    os.makedirs('./robust_results', exist_ok=True)
    with open(result_path['baseline'], 'w') as f:
        json.dump({'accuracy': base_acc}, f, indent=2)
    with open(result_path['ct'], 'w') as f:
        json.dump({'best_accuracy': best_acc, 'best_beta': best_beta, 'accuracy_list': acc_list}, f, indent=2)


if __name__ == '__main__':
    main()
