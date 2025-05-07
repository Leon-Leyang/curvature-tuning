"""
This script evaluates the generalization improvement achieved by LoRA with CT across various image classification datasets.
"""
import copy

import torch
import numpy as np
from torch import nn as nn
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta, count_trainable_parameters
from utils.curvature_tuning import replace_module, CT
from utils.lora import get_lora_model
import torch.optim as optim
from train import train_epoch, test_epoch
from loguru import logger
import argparse
import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def replace_and_transfer_with_lora(model_name, pretrained_ds, transfer_ds, rank, alpha, epochs, train_loader, val_loader):
    """
    Transfer learning.
    """
    logger.info(f"Transfer learning from {pretrained_ds} to {transfer_ds}")
    wandb.init(project='curvature-tuning', entity='leyang_hu')

    # Get the LoRA version of the model
    model = get_pretrained_model(pretrained_ds, model_name)
    model = get_lora_model(model, r=rank, alpha=alpha)
    num_classes = DATASET_TO_NUM_CLASSES[transfer_ds]
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    model = model.to(device)

    # Count how many parameters are trainable
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"LoRA: rank = {rank}, alpha = {alpha}")
    logger.debug(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_loss = float('inf')
    os.makedirs('./ckpts', exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler=None)
        val_loss, _ = test_epoch(epoch, model, val_loader, criterion, device)
        if val_loss < best_loss:
            logger.debug(f'New best validation loss: {val_loss} at epoch {epoch}')
            best_loss = val_loss
            torch.save(model.state_dict(), f'./ckpts/{model_name}_rank{rank}_alpha{alpha}_{pretrained_ds}_to_{transfer_ds}_best.pth')

    model.load_state_dict(torch.load(f'./ckpts/{model_name}_rank{rank}_alpha{alpha}_{pretrained_ds}_to_{transfer_ds}_best.pth'))

    wandb.finish()

    return model


def replace_then_lora_test_acc(beta_vals, pretrained_ds, transfer_ds, model_name='resnet18',  rank=1, alpha=1.0, epochs=30):
    """
    Replace ReLU with CT and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'
    train_loader, test_loader, val_loader = get_data_loaders(dataset, val_size=-1, train_batch_size=300)

    logger.info(f'Running replace then lora accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    val_acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = replace_and_transfer_with_lora(model_name, pretrained_ds, transfer_ds, rank, alpha, epochs, train_loader, val_loader)
    _, relu_val_acc = test_epoch(-1, transfer_model, val_loader, criterion, device)
    best_val_acc = relu_val_acc
    best_val_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(transfer_model), nn.ReLU, CT, beta=beta)
        _, val_acc = test_epoch(-1, new_model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_beta = beta
        val_acc_list.append(val_acc)
        beta_list.append(beta)
    val_acc_list.append(relu_val_acc)
    beta_list.append(1)

    plot_metric_vs_beta(val_acc_list, beta_list, relu_val_acc, dataset, metric='Val Accuracy')

    logger.debug(f'Testing ReLU')
    _, relu_test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)

    if best_val_beta != 1:
        logger.debug(f'Testing best CT with beta={best_val_beta:.2f}')
        new_model = replace_module(copy.deepcopy(transfer_model), nn.ReLU, CT, beta=best_val_beta)
        _, best_test_acc = test_epoch(-1, new_model, test_loader, criterion, device)
    else:
        logger.debug(f'Skipping testing best CT as beta=1')
        best_test_acc = relu_test_acc

    logger.info(
        f'Best accuracy for {dataset}: {best_test_acc:.2f} with beta={best_val_beta:.2f}, compared to ReLU accuracy: {relu_test_acc:.2f}')
    logger.info(f'Best validation accuracy for {dataset}: {best_val_acc:.2f} with beta={best_val_beta:.2f}, compared to ReLU validation accuracy: {relu_val_acc:.2f}')


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on image classification datasets (test set unseen)')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--rank', type=int, default=1, help='Rank for LoRA')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for LoRA')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrained_ds', type=str, nargs='+',
                        default=['imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['arabic_characters', 'beans', 'fgvc_aircraft'], help='List of transfer datasets')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_rank{args.rank}_alpha{args.alpha}_epoch{args.epochs}_{args.model}_seed{args.seed}')
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

                replace_then_lora_test_acc(betas, pretrained_ds, transfer_ds, args.model, args.rank, args.alpha, args.epochs)


if __name__ == '__main__':
    main()
