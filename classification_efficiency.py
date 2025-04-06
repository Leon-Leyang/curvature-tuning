"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
import numpy as np
from torch import nn as nn
from utils.data import get_data_loaders
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta
from utils.curvature_tuning import replace_module, CT
from train import test_epoch
from loguru import logger
import copy
import argparse
from classification_test_unseen import transfer_linear_probe

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def replace_then_lp_test_acc(beta_vals, pretrained_ds, transfer_ds, reg=1, coeff=0.5, topk=1, model_name='resnet18', train_size=10000, use_gd=False):
    """
    Replace ReLU with CT and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds, model_name)

    model_name = model.__class__.__name__

    train_loader, test_loader, val_loader = get_data_loaders(dataset, val_size=-1)
    beta_train_loader, _, _ = get_data_loaders(dataset, train_size=train_size, val_size=-1)

    logger.info(f'Running replace then linear probe accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, reg, topk, beta_train_loader, use_gd)
    _, relu_val_acc = test_epoch(-1, transfer_model, val_loader, criterion, device)
    best_val_acc = relu_val_acc
    best_val_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, reg, topk, beta_train_loader, use_gd)
        _, val_acc = test_epoch(-1, transfer_model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_beta = beta

    logger.debug(f'Testing ReLU')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, reg, topk, train_loader, use_gd)
    _, relu_test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)

    if best_val_beta != 1:
        logger.debug(f'Testing best CT with beta={best_val_beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=best_val_beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, reg, topk, train_loader, use_gd)
        _, best_test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
    else:
        logger.debug(f'Skipping testing best CT as beta=1')
        best_test_acc = relu_test_acc

    logger.info(
        f'Best accuracy for {dataset}: {best_test_acc:.2f} with beta={best_val_beta:.2f}, compared to ReLU accuracy: {relu_test_acc:.2f}')
    logger.info(f'Best validation accuracy for {dataset}: {best_val_acc:.2f} with beta={best_val_beta:.2f}, compared to ReLU validation accuracy: {relu_val_acc:.2f}')


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on image classification datasets (test set unseen, reduced train set used for beta search')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--coeff', type=float, default=0.5, help='Coefficient for CT')
    parser.add_argument('--use_gd', action='store_true', help='Use gradient descent to train the linear layer')
    parser.add_argument('--reg', type=float, default=1, help='Regularization strength for Logistic Regression')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topk', type=int, default=1, help='Number of top-k feature layers to use')
    parser.add_argument('--pretrained_ds', type=str, nargs='+',
                        default=['imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['arabic_characters', 'beans', 'fgvc_aircraft'], help='List of transfer datasets')
    parser.add_argument('--train_percentage', type=float, default=0.1, help='Percentage of training data to use during beta search')
    return parser.parse_args()


def main():
    args = get_args()

    use_gd = args.use_gd
    if args.topk > 1:
        use_gd = True

    f_name = get_file_name(__file__)
    if not use_gd:
        log_file_path = set_logger(
            name=f'{f_name}_train_percentage{args.train_percentage}_coeff{args.coeff}_topk{args.topk}_reg{args.reg}_{args.model}_seed{args.seed}')
    else:
        log_file_path = set_logger(
            name=f'{f_name}_train_percentage{args.train_percentage}_coeff{args.coeff}_topk{args.topk}_gd_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    betas = np.arange(0.5, 1 - 1e-6, 0.01)

    pretrained_datasets = args.pretrained_ds
    transfer_datasets = args.transfer_ds

    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            fix_seed(args.seed)  # Fix the seed each time

            if pretrained_ds == transfer_ds:  # Test on the same dataset
                raise ValueError('Currently not supporting testing on the same dataset since the test set shall not be split for validation.')

            else:  # Test on different datasets
                if result_exists(f'{pretrained_ds}_to_{transfer_ds}'):
                    logger.info(f'Skipping {pretrained_ds} to {transfer_ds} as result already exists.')
                    continue

                # Hack to get the full training set size after splitting validation
                train_loader, test_loader, val_loader = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}', val_size=-1)
                full_train_size = len(train_loader.dataset)
                train_size = int(args.train_percentage * full_train_size)
                logger.debug(f'Full train size: {full_train_size}, Beta search train size: {train_size}')

                replace_then_lp_test_acc(betas, pretrained_ds, transfer_ds, args.reg, args.coeff, args.topk, args.model, train_size, use_gd)


if __name__ == '__main__':
    main()
