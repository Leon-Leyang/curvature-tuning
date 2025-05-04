"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger
from utils.curvature_tuning import CT, replace_module_per_channel, get_mean_beta_and_coeff
from utils.lora import get_lora_cnn
from train import train_epoch, test_epoch, WarmUpLR
from loguru import logger
import copy
import argparse
import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
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

    os.makedirs('./ckpts', exist_ok=True)

    f_name = get_file_name(__file__)
    transfer_ds_alias = args.transfer_ds.replace('/', '-')
    log_file_path = set_logger(
        name=f'{f_name}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    fix_seed(args.seed)  # Fix the seed each time

    dataset = f'{args.pretrained_ds}_to_{args.transfer_ds}'

    # Freeze the backbone model and replace the last layer
    model = get_pretrained_model(args.pretrained_ds, args.model)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)

    train_loader, test_loader, val_loader = get_data_loaders(dataset, args.seed)

    criterion = nn.CrossEntropyLoss()

    # Test the baseline model
    logger.info('Testing baseline...')
    identifier = f'base_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    relu_model = copy.deepcopy(model)
    num_params_base = sum(param.numel() for param in relu_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_base}')
    logger.info(f'Starting transfer learning...')
    relu_model = transfer(relu_model, train_loader, val_loader)
    _, relu_acc = test_epoch(-1, relu_model, test_loader, criterion, device)
    logger.info(f'Baseline Accuracy: {relu_acc:.2f}')
    wandb.finish()

    # Test the model with CT
    logger.info(f'Testing CT...')
    identifier = f'ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    dummy_input_shape = (1, 3, 224, 224)
    ct_model = replace_module_per_channel(copy.deepcopy(model), dummy_input_shape, old_module=nn.ReLU,
                                          new_module=CT).to(device)
    num_params_ct = sum(param.numel() for param in ct_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_ct}')
    logger.info(f'Starting transfer learning...')
    ct_model = transfer(ct_model, train_loader, val_loader)
    _, ct_acc = test_epoch(-1, ct_model, test_loader, criterion, device)
    logger.info(f'CT Accuracy: {ct_acc:.2f}')
    wandb.finish()

    # Save the CT model
    torch.save(ct_model.state_dict(), f'./ckpts/{f_name}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'CT model saved to ./ckpts/{f_name}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')

    # Test the model with LoRA
    logger.info(f'Testing LoRA...')
    identifier = f'lora_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    lora_model = get_lora_cnn(copy.deepcopy(model), r=1, alpha=1).to(device)
    num_params_lora = sum(param.numel() for param in lora_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_lora}')
    logger.info(f'Starting transfer learning...')
    lora_model = transfer(lora_model, train_loader, val_loader)
    _, lora_acc = test_epoch(-1, lora_model, test_loader, criterion, device)
    logger.info(f'LoRA Accuracy: {lora_acc:.2f}')
    wandb.finish()

    rel_improve_base = (ct_acc - relu_acc) / relu_acc
    rel_improve_lora = (ct_acc - lora_acc) / lora_acc
    logger.info(f'Relative accuracy improvement over baseline: {rel_improve_base * 100:.2f}%')
    logger.info(f'Relative accuracy improvement over LoRA: {rel_improve_lora * 100:.2f}%')
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')


if __name__ == '__main__':
    main()
