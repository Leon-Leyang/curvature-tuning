"""
This file is for counting the number of trainable parameters in a ResNet with or without LoRA.
"""
from utils.lora import get_lora_cnn
from utils.utils import count_trainable_parameters, get_pretrained_model, get_file_name, fix_seed, set_logger
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
import argparse
from loguru import logger
import wandb
import torch.nn as nn
from train import train_epoch, test_epoch
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer_with_lora(model, pretrained_ds, transfer_ds, rank, alpha, epochs):
    logger.info(f"Transfer learning from {pretrained_ds} to {transfer_ds}")
    wandb.init(project='curvature-tuning', entity='leyang_hu')

    dataset = f'{pretrained_ds}_to_{transfer_ds}'
    train_loader, test_loader, _ = get_data_loaders(dataset, train_batch_size=1000)

    # Get the LoRA version of the model
    model = get_pretrained_model(pretrained_ds, model)
    model = get_lora_cnn(model, r=rank, alpha=alpha)
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

    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler=None)

    test_epoch(-1, model, test_loader, criterion, device)

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on image classification datasets with LoRA')
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
                        default=['arabic_characters', 'arabic_digits', 'beans', 'cub200', 'dtd', 'fashion_mnist', 'fgvc_aircraft', 'flowers102', 'food101'], help='List of transfer datasets')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_rank{args.rank}_alpha{args.alpha}_epoch{args.epochs}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    pretrained_ds = args.pretrained_ds
    transfer_ds = args.transfer_ds

    rank = args.rank
    alpha = args.alpha
    epochs = args.epochs

    for pretrained_ds in pretrained_ds:
        for transfer_ds in transfer_ds:
            fix_seed(args.seed)
            transfer_with_lora(args.model, pretrained_ds, transfer_ds, rank, alpha, epochs)


if __name__ == "__main__":
    main()
