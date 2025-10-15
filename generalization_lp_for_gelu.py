"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import replace_module, CTU
from train import test_epoch, linear_probe
from loguru import logger
import copy
import argparse
import wandb
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on image classification datasets')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrained_ds', type=str, default='imagenet', help='Pretrained dataset')
    parser.add_argument('--transfer_ds', type=str, default='beans', help='Transfer dataset')
    parser.add_argument('--linear_probe_train_bs', type=int, default=32, help='Batch size for linear probe')
    parser.add_argument('--linear_probe_test_bs', type=int, default=800, help='Batch size for linear probe test')
    return parser.parse_args()


def main():
    args = get_args()

    transfer_ds_alias = args.transfer_ds.replace('/', '-')

    result_path = {
        'baseline': f'./results/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
    }

    # Check if all result files exist
    if all(os.path.exists(path) for path in result_path.values()):
        print('All result files already exist. Exiting...')
        return

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    fix_seed(args.seed)  # Fix the seed each time

    logger.info(f'Running on {device}')

    dataset = f'{args.pretrained_ds}_to_{args.transfer_ds}'

    # Freeze the backbone model and replace the last layer
    model = get_pretrained_model(args.pretrained_ds, args.model)
    for param in model.parameters():
        param.requires_grad = False
    if 'swin' in args.model:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    elif 'vgg' in args.model:
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed, train_batch_size=args.linear_probe_train_bs,
                                                              test_batch_size=args.linear_probe_test_bs)

    criterion = nn.CrossEntropyLoss()

    # Test the baseline model
    identifier = f'base_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        name=identifier,
        config=vars(args),
    )
    logger.info('Testing baseline...')
    base_model = copy.deepcopy(model)
    num_params_base = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_base}')
    logger.info(f'Starting transfer learning...')
    base_model, _ = linear_probe(base_model, train_loader, val_loader, new_train_batch_size=args.linear_probe_train_bs, new_val_batch_size=args.linear_probe_test_bs)
    _, base_acc = test_epoch(-1, base_model, test_loader, criterion, device)
    logger.info(f'Baseline Accuracy: {base_acc:.2f}%')
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(base_model.state_dict(), f'./ckpts/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'Baseline model saved to ./ckpts/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': base_acc, 'num_params': num_params_base})
    wandb.finish()

    # Log the summary
    logger.info(f'Baseline Accuracy: {base_acc:.2f}%')

    # Save the results
    os.makedirs('./results', exist_ok=True)
    save_result_json(
        result_path['baseline'],
        num_params_base, base_acc)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
