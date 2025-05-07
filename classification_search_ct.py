"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import replace_module, SharedCT
from train import test_epoch, linear_probe
from loguru import logger
import copy
import argparse
import wandb
import os
import time
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

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed, train_batch_size=args.linear_probe_train_bs,
                                                              test_batch_size=args.linear_probe_test_bs)

    criterion = nn.CrossEntropyLoss()

    beta_range = np.arange(0.7, 1.0 - 1e-6, 0.01)

    identifier = f'search_ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    # Associate the subplots with the same beta
    for beta in beta_range:
        wandb.define_metric(f"epoch_beta{beta:.2f}")
        wandb.define_metric(f"train_loss_beta{beta:.2f}", step_metric=f"epoch_beta{beta:.2f}")
        wandb.define_metric(f"train_accuracy_beta{beta:.2f}", step_metric=f"epoch_beta{beta:.2f}")
        wandb.define_metric(f"val_loss_beta{beta:.2f}", step_metric=f"epoch_beta{beta:.2f}")
        wandb.define_metric(f"val_accuracy_beta{beta:.2f}", step_metric=f"epoch_beta{beta:.2f}")
        wandb.define_metric(f"lr_beta{beta:.2f}", step_metric=f"epoch_beta{beta:.2f}")

    # Search for the best beta
    best_val_acc = 0.0
    best_beta = None
    best_model = None
    val_acc_list = []
    start_time = time.perf_counter()
    for beta in beta_range:
        logger.info(f'Testing Search CT with beta: {beta:.2f}')
        shared_raw_beta = nn.Parameter(torch.logit(torch.tensor(beta)), requires_grad=False)
        shared_raw_coeff = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        ct_model = replace_module(copy.deepcopy(model), old_module=nn.ReLU, new_module=SharedCT,
                                  shared_raw_beta=shared_raw_beta, shared_raw_coeff=shared_raw_coeff).to(device)
        num_params_ct = sum(param.numel() for param in ct_model.parameters() if param.requires_grad)
        logger.info(f'Number of trainable parameters: {num_params_ct}')
        logger.info(f'Starting transfer learning...')
        ct_model, val_acc = linear_probe(ct_model, train_loader, val_loader, beta,
                                         new_train_batch_size=args.linear_probe_train_bs,
                                         new_val_batch_size=args.linear_probe_test_bs)
        logger.info(f'Best validation accuracy for beta {beta:.2f}: {val_acc:.2f}%')

        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_model = copy.deepcopy(ct_model)
            best_val_acc = val_acc
            best_beta = beta

    end_time = time.perf_counter()
    ct_transfer_time = int(end_time - start_time)
    logger.info(f'Search CT Transfer learning time: {ct_transfer_time} seconds')

    logger.info(f'Best beta: {best_beta:.2f}, Best validation accuracy: {best_val_acc:.2f}%')

    logger.info('Testing the best model...')
    start_time = time.perf_counter()
    _, test_acc = test_epoch(-1, best_model, test_loader, criterion, device, best_beta)
    end_time = time.perf_counter()
    ct_test_time = int(end_time - start_time)
    logger.info(f'Search CT Test time: {ct_test_time} seconds')
    logger.info(f'Test accuracy: {test_acc:.2f}%')

    # Save the Search CT model
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(best_model.state_dict(), f'./ckpts/search_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'Search CT model saved to ./ckpts/search_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': test_acc, 'transfer_time': ct_transfer_time, 'test_time': ct_test_time, 'num_params': num_params_ct, 'best_beta': best_beta})
    wandb.finish()

    # Log the summary
    logger.info(f'Search CT model trainable parameters: {num_params_ct}')
    logger.info(f'Search CT Transfer learning time: {ct_transfer_time} seconds, Test time: {ct_test_time} seconds')
    logger.info(f'Best beta: {best_beta:.2f}, Best validation accuracy: {best_val_acc:.2f}%')
    logger.info(f'Final Accuracy: {test_acc:.2f}%')

    # Save the results
    os.makedirs('./results', exist_ok=True)
    save_result_json(
        f'./results/search_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        num_params_ct, test_acc, ct_transfer_time, ct_test_time, beta=best_beta, coeff=0.5, best_val_acc=best_val_acc,
        val_acc_list=val_acc_list)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
