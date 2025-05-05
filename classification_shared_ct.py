"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import replace_module, SharedCT
from train import train_epoch, test_epoch, WarmUpLR
from loguru import logger
import copy
import argparse
import wandb
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()

    ct_params = []
    other_params = []

    for module in model.modules():
        if isinstance(module, SharedCT):
            ct_params += [p for p in module.parameters() if p.requires_grad]
        else:
            other_params += [p for p in module.parameters() if p.requires_grad]

    # Avoid duplicates since the search is done in a nested loop
    ct_param_set = set(ct_params)
    other_params = [p for p in other_params if p not in ct_param_set]

    optimizer = torch.optim.Adam([
        {'params': ct_params, 'lr': 1e-1},
        {'params': other_params, 'lr': 1e-3}
    ])

    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    best_model = None
    best_acc = 0.0

    for epoch in range(1, 31):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler)
        _, val_acc = test_epoch(epoch, model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = val_acc
            logger.info(f'New best validation accuracy: {val_acc:.2f} at epoch {epoch}')
        scheduler.step()
    return best_model


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
    parser.add_argument('--transfer_ds', type=str, default='cifar10', help='Transfer dataset')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    transfer_ds_alias = args.transfer_ds.replace('/', '-')
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
    if 'swin' not in args.model:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    else:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)

    train_loader, test_loader, val_loader = get_data_loaders(dataset, args.seed)

    criterion = nn.CrossEntropyLoss()

    # Test the model with CT
    identifier = f'shared_ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    logger.info(f'Testing Shared CT...')
    shared_raw_beta = nn.Parameter(torch.tensor(1.386))
    shared_raw_coeff = nn.Parameter(torch.tensor(0.0))
    ct_model = replace_module(copy.deepcopy(model), old_module=nn.ReLU, new_module=SharedCT,
                              shared_raw_beta=shared_raw_beta, shared_raw_coeff=shared_raw_coeff).to(device)
    num_params_ct = sum(param.numel() for param in ct_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_ct}')
    beta = torch.sigmoid(shared_raw_beta).item()
    coeff = torch.sigmoid(shared_raw_coeff).item()
    logger.info(f'Beta: {beta:.6f}, Coeff: {coeff:.6f}')
    logger.info(f'Starting transfer learning...')
    start_time = time.perf_counter()
    ct_model = transfer(ct_model, train_loader, val_loader)
    end_time = time.perf_counter()
    ct_transfer_time = int(end_time - start_time)
    logger.info(f'Shared CT Transfer learning time: {ct_transfer_time} seconds')
    start_time = time.perf_counter()
    _, ct_acc = test_epoch(-1, ct_model, test_loader, criterion, device)
    end_time = time.perf_counter()
    ct_test_time = int(end_time - start_time)
    logger.info(f'Shared CT Test time: {ct_test_time} seconds')
    logger.info(f'Shared CT Accuracy: {ct_acc:.2f}%')

    beta = torch.sigmoid(shared_raw_beta).item()
    coeff = torch.sigmoid(shared_raw_coeff).item()
    logger.info(f'Beta: {beta:.6f}, Coeff: {coeff:.6f}')

    # Save the Shared CT model
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(ct_model.state_dict(), f'./ckpts/shared_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'Shared CT model saved to ./ckpts/shared_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': ct_acc, 'transfer_time': ct_transfer_time, 'test_time': ct_test_time, 'num_params': num_params_ct})
    wandb.finish()

    # Log the summary
    logger.info(f'Shared CT model trainable parameters: {num_params_ct}')
    logger.info(f'Shared CT Transfer learning time: {ct_transfer_time} seconds, Test time: {ct_test_time} seconds')
    logger.info(f'Shared CT Accuracy: {ct_acc:.2f}%')
    beta = torch.sigmoid(shared_raw_beta).item()
    coeff = torch.sigmoid(shared_raw_coeff).item()
    logger.info(f'Beta: {beta:.6f}, Coeff: {coeff:.6f}')

    # Save the results
    os.makedirs('./results', exist_ok=True)
    save_result_json(
        f'./results/shared_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        num_params_ct, ct_acc, ct_transfer_time, ct_test_time, beta=beta, coeff=coeff)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
