"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets,
now with LR cross-validation over {1e-2, 1e-3, 1e-4} for the linear probe stage. The LR that yields the best
validation accuracy is chosen to evaluate on the test set. We also log per-LR validation accuracies to W&B,
and record `best_lr` in the saved results JSON.
"""
import torch
from torch import nn as nn
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from train import test_epoch, linear_probe
from loguru import logger
import copy
import argparse
import wandb
import os

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
        model.head = nn.Linear(in_features=model.head.in_features,
                               out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    elif 'vgg' in args.model:
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features,
                                         out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features,
                             out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]).to(device)

    train_loader, test_loader, val_loader = get_data_loaders(
        dataset,
        seed=args.seed,
        train_batch_size=args.linear_probe_train_bs,
        test_batch_size=args.linear_probe_test_bs
    )

    criterion = nn.CrossEntropyLoss()

    # Learning rate cross-validation
    logger.info('Testing baseline with LR cross-validation...')
    tmp_model_for_count = copy.deepcopy(model)
    num_params_base = sum(p.numel() for p in tmp_model_for_count.parameters() if p.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_base}')
    del tmp_model_for_count

    candidate_lrs = [1e-2, 1e-3, 1e-4]
    lr_to_val = {}
    lr_to_model = {}

    for lr in candidate_lrs:
        run_name = f'base_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}_lr{lr:.0e}'
        wandb.init(
            project='ct-new',
            name=run_name,
            config={**vars(args), 'lr': lr},
            reinit=True,  # allow multiple runs in a single script
        )

        logger.info(f'Starting linear probe with lr={lr:.0e}...')
        m = copy.deepcopy(model)
        m, val_acc = linear_probe(
            m,
            train_loader,
            val_loader,
            new_train_batch_size=args.linear_probe_train_bs,
            new_val_batch_size=args.linear_probe_test_bs,
            lr=lr
        )
        lr_to_val[lr] = float(val_acc)
        lr_to_model[lr] = m

        logger.info(f'Validation Accuracy (lr={lr:.0e}): {val_acc:.2f}%')
        wandb.log({'val_accuracy': val_acc, 'num_params': num_params_base})
        wandb.finish()

    # Select best lr by validation accuracy
    best_lr = max(lr_to_val, key=lr_to_val.get)
    best_val_acc = lr_to_val[best_lr]
    best_model = lr_to_model[best_lr]
    logger.info(f'Best lr={best_lr:.0e} with val_acc={best_val_acc:.2f}%')

    # Evaluate best model on test set
    logger.info('Evaluating best model on test set...')
    _, base_acc = test_epoch(-1, best_model, test_loader, criterion, device)
    logger.info(f'Baseline Test Accuracy (best lr): {base_acc:.2f}%')

    # Log final summary as its own W&B run
    summary_name = f'base_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(project='ct-new', name=summary_name, config=vars(args), reinit=True)
    wandb.log({
        'test_accuracy': base_acc,
        'best_lr': best_lr,
        'best_val_accuracy': best_val_acc,
        'num_params': num_params_base,
        **{f'val_accuracy_lr_{lr:.0e}': acc for lr, acc in lr_to_val.items()}
    })
    wandb.finish()

    # Save checkpoint for the best model
    os.makedirs('./ckpts', exist_ok=True)
    ckpt_path = f'./ckpts/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth'
    torch.save(best_model.state_dict(), ckpt_path)
    logger.info(f'Best baseline model saved to {ckpt_path}')

    # Save the results
    os.makedirs('./results', exist_ok=True)
    save_result_json(
        result_path['baseline'],
        num_params_base,
        base_acc,
        best_lr=best_lr
    )
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
