"""
This script evaluates the robustness improvement achieved by CT across various datasets and attacks.
"""
import os
import torch
from pathlib import Path
from torch import nn as nn
from utils.robustbench import benchmark
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger
from utils.data import DATASET_TO_NUM_CLASSES, get_data_loaders
from utils.curvature_tuning import TrainableCTU, replace_module_dynamic, get_mean_beta_and_coeff
from utils.lora import get_lora_model
from generalization_trainable_ct import transfer
from loguru import logger
from robustness import get_transform, THREAT_TO_EPS
import copy
import argparse
import json
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Robustness experiments on RobustBench')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='Dataset on which to test the model')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for robustness tests')
    parser.add_argument('--n_examples', type=int, default=1000, help='Number of samples to test')
    parser.add_argument('--transfer_train_bs', type=int, default=32, help='Batch size for transfer learning')
    parser.add_argument('--transfer_test_bs', type=int, default=800, help='Batch size for transfer learning test')
    return parser.parse_args()


def main():
    args = get_args()

    lora_rank = 1
    lora_alpha = lora_rank

    dataset = f'imagenet_to_{args.dataset}'

    result_path = {
        'Linf':
            {'baseline': f'./robust_results/base_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
            'train_ct': f'./robust_results/train_ct_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
            'lora': f'./robust_results/lora_rank{lora_rank}_alpha{lora_alpha}_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json'},
        # 'L2':
        #     {'baseline': f'./robust_results/base_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
        #     'train_ct': f'./robust_results/train_ct_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
        #     'lora': f'./robust_results/lora_rank{lora_rank}_alpha{lora_alpha}_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json'},
        # 'corruptions':
        #     {'baseline': f'./robust_results/base_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
        #     'train_ct': f'./robust_results/train_ct_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json',
        #     'lora': f'./robust_results/lora_rank{lora_rank}_alpha{lora_alpha}_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json'},
    }

    state_path = {
        'Linf':
            {'baseline': Path(f"./cache/base_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
            'train_ct': Path(f"./cache/train_ct_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
            'lora': Path(f"./cache/lora_rank{lora_rank}_alpha{lora_alpha}_linf_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json")},
        # 'L2':
        #     {'baseline': Path(f"./cache/base_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
        #     'train_ct': Path(f"./cache/train_ct_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
        #     'lora': Path(f"./cache/lora_rank{lora_rank}_alpha{lora_alpha}_l2_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json")},
        # 'corruptions':
        #     {'baseline': Path(f"./cache/base_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
        #     'train_ct': Path(f"./cache/train_ct_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json"),
        #     'lora': Path(f"./cache/lora_rank{lora_rank}_alpha{lora_alpha}_corruptions_{dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}.json")},
    }

    # Check if all paths exist
    all_exist = all(
        os.path.exists(path)
        for threat_dict in result_path.values()
        for path in threat_dict.values()
    )
    if all_exist:
        print('All result files already exist. Exiting...')
        return

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_{args.dataset}_sample{args.n_examples}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    fix_seed(args.seed)  # Fix the seed each time

    logger.info(f'Running on {device}')

    model = get_pretrained_model('imagenet', args.model)
    for param in model.parameters():
        param.requires_grad = False
    if 'swin' in args.model:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)
    elif 'vgg' in args.model:
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)

    model.eval()

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed, train_batch_size=args.transfer_train_bs, test_batch_size=args.transfer_test_bs)

    # Get the transferred baseline model
    logger.info('Transferring the baseline model...')
    identifier = f'base_{dataset}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        name=identifier,
        config=vars(args),
    )
    base_model = transfer(copy.deepcopy(model), train_loader, val_loader)
    wandb.finish()

    # Get the model transferred with trainable CT
    logger.info('Transferring the model with Trainable CT...')
    identifier = f'train_ct_{dataset}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        name=identifier,
        config=vars(args),
    )
    dummy_input_shape = (1, 3, 224, 224)
    ct_model = replace_module_dynamic(copy.deepcopy(model), dummy_input_shape, old_module=nn.ReLU,
                                      new_module=TrainableCTU).to(device)
    ct_model = transfer(ct_model, train_loader, val_loader)
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')
    wandb.finish()

    # Get the model transferred with LoRA
    logger.info('Transferring the model with LoRA...')
    identifier = f'lora_rank{lora_rank}_alpha{lora_alpha}_{dataset}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        name=identifier,
        config=vars(args),
    )
    lora_model = get_lora_model(copy.deepcopy(model), r=lora_rank, alpha=lora_alpha).to(device)
    # Replace the last layer with normal linear layer
    if 'swin' in args.model:
        lora_model.head = nn.Linear(in_features=lora_model.head.in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)
    elif 'vgg' in args.model:
        lora_model.classifier[-1] = nn.Linear(in_features=lora_model.classifier[-1].in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)
    else:
        lora_model.fc = nn.Linear(in_features=lora_model.fc.in_features, out_features=DATASET_TO_NUM_CLASSES[args.dataset]).to(device)
    lora_model = transfer(lora_model, train_loader, val_loader, lr=1e-4)
    wandb.finish()

    # Make directory for evaluation cache
    os.makedirs('./cache', exist_ok=True)

    data_dir = './data/imagenet' if 'imagenet' in args.dataset else './data'

    # for threat in ["Linf", "L2", "corruptions"]:
    for threat in ["Linf"]:
        transform = get_transform(threat, args.dataset)

        # Test the baseline model
        logger.info(f'Testing the baseline on {threat}...')
        _, base_acc = benchmark(
            copy.deepcopy(base_model), dataset=args.dataset, threat_model=threat, eps=THREAT_TO_EPS[threat], device=device,
            batch_size=args.batch_size, preprocessing=transform, n_examples=args.n_examples,
            aa_state_path=state_path[threat]['baseline'], seed=args.seed, data_dir=data_dir
        )
        base_acc *= 100
        logger.info(f'Baseline accuracy: {base_acc:.2f}%')

        # Test the model with Trainable CT
        logger.info(f'Testing Trainable CT on {threat}...')
        _, ct_acc = benchmark(
            copy.deepcopy(ct_model), dataset=args.dataset, threat_model=threat, eps=THREAT_TO_EPS[threat], device=device,
            batch_size=args.batch_size, preprocessing=transform, n_examples=args.n_examples,
            aa_state_path=state_path[threat]['train_ct'], seed=args.seed, data_dir=data_dir
        )
        ct_acc *= 100
        logger.info(f'Trainable CT accuracy: {ct_acc:.2f}%')

        # Test the model with LoRA
        logger.info(f'Testing LoRA on {threat}...')
        _, lora_acc = benchmark(
            copy.deepcopy(lora_model), dataset=args.dataset, threat_model=threat, eps=THREAT_TO_EPS[threat], device=device,
            batch_size=args.batch_size, preprocessing=transform, n_examples=args.n_examples,
            aa_state_path=state_path[threat]['lora'], seed=args.seed, data_dir=data_dir
        )
        lora_acc *= 100
        logger.info(f'LoRA accuracy: {lora_acc:.2f}%')

        with open(result_path[threat]['baseline'], 'w') as f:
            json.dump({'accuracy': base_acc}, f, indent=2)
        with open(result_path[threat]['train_ct'], 'w') as f:
            json.dump({'accuracy': ct_acc, 'beta': mean_beta, 'coeff': mean_coeff}, f, indent=2)
        with open(result_path[threat]['lora'], 'w') as f:
            json.dump({'accuracy': lora_acc}, f, indent=2)


if __name__ == '__main__':
    main()
