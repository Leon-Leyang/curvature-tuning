"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import CT, replace_module_per_channel, get_mean_beta_and_coeff
from utils.lora import get_lora_cnn
from train import train_epoch, test_epoch, WarmUpLR, linear_probe
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
        if isinstance(module, CT):
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

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed, train_batch_size=32, test_batch_size=800)

    criterion = nn.CrossEntropyLoss()

    # Test the baseline model
    identifier = f'base_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    logger.info('Testing baseline...')
    base_model = copy.deepcopy(model)
    num_params_base = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_base}')
    logger.info(f'Starting transfer learning...')
    start_time = time.perf_counter()
    base_model, _ = linear_probe(base_model, train_loader, val_loader)
    end_time = time.perf_counter()
    base_transfer_time = int(end_time - start_time)
    logger.info(f'Baseline Transfer learning time: {base_transfer_time} seconds')
    start_time = time.perf_counter()
    _, base_acc = test_epoch(-1, base_model, test_loader, criterion, device)
    end_time = time.perf_counter()
    base_test_time = int(end_time - start_time)
    logger.info(f'Baseline Test time: {base_test_time} seconds')
    logger.info(f'Baseline Accuracy: {base_acc:.2f}%')
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(base_model.state_dict(), f'./ckpts/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'Baseline model saved to ./ckpts/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': base_acc, 'transfer_time': base_transfer_time, 'test_time': base_test_time, 'num_params': num_params_base})
    wandb.finish()

    # Test the model with CT
    identifier = f'ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    logger.info(f'Testing CT...')
    dummy_input_shape = (1, 3, 224, 224)
    ct_model = replace_module_per_channel(copy.deepcopy(model), dummy_input_shape, old_module=nn.ReLU,
                                          new_module=CT).to(device)
    num_params_ct = sum(param.numel() for param in ct_model.parameters() if param.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params_ct}')
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')
    logger.info(f'Starting transfer learning...')
    start_time = time.perf_counter()
    ct_model = transfer(ct_model, train_loader, val_loader)
    end_time = time.perf_counter()
    ct_transfer_time = int(end_time - start_time)
    logger.info(f'CT Transfer learning time: {ct_transfer_time} seconds')
    start_time = time.perf_counter()
    _, ct_acc = test_epoch(-1, ct_model, test_loader, criterion, device)
    end_time = time.perf_counter()
    ct_test_time = int(end_time - start_time)
    logger.info(f'CT Test time: {ct_test_time} seconds')
    logger.info(f'CT Accuracy: {ct_acc:.2f}%')

    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')

    # Save the CT model
    torch.save(ct_model.state_dict(), f'./ckpts/ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'CT model saved to ./ckpts/ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': ct_acc, 'transfer_time': ct_transfer_time, 'test_time': ct_test_time, 'num_params': num_params_ct})
    wandb.finish()

    # Test the model with LoRA
    lora_rank = 1
    lora_alpha = lora_rank
    identifier = f'lora_rank{lora_rank}_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct-new',
        entity='leyang_hu',
        name=identifier,
        config=vars(args),
    )
    logger.info(f'Testing LoRA...')
    lora_model = get_lora_cnn(copy.deepcopy(model), r=lora_rank, alpha=lora_alpha).to(device)
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
    start_time = time.perf_counter()
    lora_model = transfer(lora_model, train_loader, val_loader)
    end_time = time.perf_counter()
    lora_transfer_time = int(end_time - start_time)
    logger.info(f'LoRA Transfer learning time: {lora_transfer_time} seconds')
    start_time = time.perf_counter()
    _, lora_acc = test_epoch(-1, lora_model, test_loader, criterion, device)
    end_time = time.perf_counter()
    lora_test_time = int(end_time - start_time)
    logger.info(f'LoRA Test time: {lora_test_time} seconds')
    logger.info(f'LoRA Accuracy: {lora_acc:.2f}%')
    torch.save(lora_model.state_dict(), f'./ckpts/lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    logger.info(f'LoRA model saved to ./ckpts/lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth')
    wandb.log({'test_accuracy': lora_acc, 'transfer_time': lora_transfer_time, 'test_time': lora_test_time, 'num_params': num_params_lora})
    wandb.finish()

    # Log the summary
    logger.info(f'Baseline model trainable parameters: {num_params_base}')
    logger.info(f'CT model trainable parameters: {num_params_ct}')
    logger.info(f'LoRA model trainable parameters: {num_params_lora}')
    logger.info(f'CT params/LoRA params: {num_params_ct / num_params_lora:.2f}')
    if num_params_lora < num_params_ct:
        logger.warning(f'LoRA model has fewer trainable parameters than CT model: {num_params_lora} < {num_params_ct}')
    logger.info(f'Baseline Transfer learning time: {base_transfer_time} seconds, Test time: {base_test_time} seconds')
    logger.info(f'CT Transfer learning time: {ct_transfer_time} seconds, Test time: {ct_test_time} seconds')
    logger.info(f'LoRA Transfer learning time: {lora_transfer_time} seconds, Test time: {lora_test_time} seconds')
    logger.info(f'CT time/LoRA time     Transfer: {ct_transfer_time / lora_transfer_time:.2f}, Test: {ct_test_time / lora_test_time:.2f}')
    rel_improve_base = (ct_acc - base_acc) / base_acc
    rel_improve_lora = (ct_acc - lora_acc) / lora_acc
    logger.info(f'Baseline Accuracy: {base_acc:.2f}%')
    logger.info(f'CT Accuracy: {ct_acc:.2f}%')
    logger.info(f'LoRA Accuracy: {lora_acc:.2f}%')
    logger.info(f'Relative accuracy improvement over baseline: {rel_improve_base * 100:.2f}%')
    logger.info(f'Relative accuracy improvement over LoRA: {rel_improve_lora * 100:.2f}%')
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')

    # Save the results
    os.makedirs('./results', exist_ok=True)
    save_result_json(
        f'./results/base_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        num_params_base, base_acc, base_transfer_time, base_test_time)
    save_result_json(
        f'./results/ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        num_params_ct, ct_acc, ct_transfer_time, ct_test_time, beta=mean_beta, coeff=mean_coeff)
    save_result_json(
        f'./results/lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        num_params_lora, lora_acc, lora_transfer_time, lora_test_time)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
