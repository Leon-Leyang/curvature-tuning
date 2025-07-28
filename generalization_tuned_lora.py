"""
This script evaluates the generalization improvement achieved by LoRA with tuned hyperparameters across various image classification datasets.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import TrainableCTU, replace_module_dynamic, get_mean_beta_and_coeff
from utils.lora import get_lora_model
from train import train_epoch, test_epoch, WarmUpLR, linear_probe
from loguru import logger
import copy
import argparse
import wandb
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer(model, train_loader, val_loader, lr=1e-3):
    criterion = nn.CrossEntropyLoss()

    ct_params = []
    other_params = []

    for module in model.modules():
        if isinstance(module, TrainableCTU):
            ct_params += [p for p in module.parameters() if p.requires_grad]
        else:
            other_params += [p for p in module.parameters() if p.requires_grad]

    # Avoid duplicates since the search is done in a nested loop
    ct_param_set = set(ct_params)
    other_params = [p for p in other_params if p not in ct_param_set]

    optimizer = torch.optim.Adam([
        {'params': ct_params, 'lr': 1e-1},
        {'params': other_params, 'lr': lr}
    ])

    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    best_model = None
    best_acc = 0.0

    for epoch in range(1, 21):
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
    parser.add_argument('--transfer_ds', type=str, default='beans', help='Transfer dataset')
    parser.add_argument('--transfer_train_bs', type=int, default=32, help='Batch size for transfer learning')
    parser.add_argument('--transfer_test_bs', type=int, default=800, help='Batch size for transfer learning test')
    parser.add_argument('--lora_rank', type=int, default=1, help='Rank for LoRA')
    return parser.parse_args()


def main():
    args = get_args()

    lora_rank = args.lora_rank

    transfer_ds_alias = args.transfer_ds.replace('/', '-')

    result_path = {
                   'lora': f'./results/tuned_lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json'
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
    if 'swin' not in args.model:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    else:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed, train_batch_size=args.transfer_train_bs, test_batch_size=args.transfer_test_bs)

    criterion = nn.CrossEntropyLoss()

    # Test the model with LoRA
    alpha_rank_ratios = [1, 2, 4]
    lora_acc_list = []
    best_lora_acc = 0.0
    best_alpha_rank_ratio = None
    for ratio in alpha_rank_ratios:
        if ratio == 1 and lora_rank == 1:
            file_path = f'./results/lora_rank1_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json'
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    result = json.load(f)
                lora_acc = result['accuracy']
            else:
                logger.warning(f'File {file_path} does not exist. Skipping LoRA with rank 1.')
                lora_acc = 0.0
        else:
            lora_alpha = lora_rank * ratio
            identifier = f'lora_rank{lora_rank}_alpha{lora_alpha}_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
            wandb.init(
                project='ct-new',
                name=identifier,
                config=vars(args),
            )
            logger.info(f'Testing LoRA with rank {lora_rank} and alpha {lora_alpha}...')
            lora_model = get_lora_model(copy.deepcopy(model), r=lora_rank, alpha=lora_alpha).to(device)
            # Replace the last layer with normal linear layer
            if 'swin' not in args.model:
                lora_model.fc = nn.Linear(in_features=lora_model.fc.in_features,
                                          out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
            else:
                lora_model.head = nn.Linear(in_features=lora_model.head.in_features,
                                            out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
            num_params_lora = sum(param.numel() for param in lora_model.parameters() if param.requires_grad)
            logger.info(f'Number of trainable parameters: {num_params_lora}')
            logger.info(f'Starting transfer learning...')
            lora_model = transfer(lora_model, train_loader, val_loader, lr=1e-4)
            _, lora_acc = test_epoch(-1, lora_model, test_loader, criterion, device)
            logger.info(f'LoRA with rank {lora_rank} and alpha {lora_alpha} accuracy: {lora_acc:.2f}%')
            torch.save(lora_model.state_dict(), f'./ckpts/lora_rank{lora_rank}_alpha{lora_alpha}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
            logger.info(f'LoRA model saved to ./ckpts/lora_rank{lora_rank}_alpha{lora_alpha}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
            wandb.log({'test_accuracy': lora_acc, 'num_params': num_params_lora})
            wandb.finish()
        lora_acc_list.append(lora_acc)
        if lora_acc > best_lora_acc:
            best_lora_acc = lora_acc
            best_alpha_rank_ratio = ratio

    # Log the summary
    logger.info(f'LoRA model trainable parameters: {num_params_lora}')
    logger.info(f'Best LoRA Accuracy: {best_lora_acc:.2f}%')
    logger.info(f'Best LoRA alpha/rank ratio: {best_alpha_rank_ratio}')
    logger.info(f'List of LoRA accuracies: {lora_acc_list}')

    # Save the results
    os.makedirs('./results', exist_ok=True)

    save_result_json(
        result_path['lora'],
        num_params_lora, best_lora_acc, lora_acc_list=lora_acc_list, best_alpha_rank_ratio=best_alpha_rank_ratio)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
