"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import replace_module, SharedCT
from train import WarmUpLR
from loguru import logger
import copy
import argparse
import wandb
import os
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def extract_features_and_labels(feature_extractor, dataloader):
    features_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            feats = feature_extractor(x)
            feats = torch.flatten(feats, 1)
            features_list.append(feats)
            labels_list.append(y)

    return torch.cat(features_list).cpu(), torch.cat(labels_list).cpu()


def linear_probe(model, train_loader, val_loader, beta):
    """
    Linear probing by extracting features using the frozen backbone (excluding the classifier),
    then training a new linear classifier on those features.
    """
    # Strip the classification head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Extract train/val features
    train_feats, train_labels = extract_features_and_labels(feature_extractor, train_loader)
    val_feats, val_labels = extract_features_and_labels(feature_extractor, val_loader)

    # Create feature datasets
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)
    train_loader_new = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
    val_loader_new = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=6)

    # Train a linear classifier
    num_features = train_feats.shape[1]
    num_classes = train_labels.max().item() + 1
    classifier = nn.Linear(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader_new))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    best_classifier = None
    best_acc = 0.0

    for epoch in range(1, 31):
        train_epoch(epoch, classifier, train_loader_new, optimizer, criterion, device, warmup_scheduler, beta)
        _, val_acc = test_epoch(epoch, classifier, val_loader_new, criterion, device, beta)
        if val_acc > best_acc:
            best_classifier = copy.deepcopy(classifier)
            best_acc = val_acc
            logger.info(f'New best validation accuracy: {val_acc:.2f} at epoch {epoch}')
        scheduler.step()

    # Replace the classifier in the original model with the trained one
    if hasattr(model, 'fc'):
        model.fc = best_classifier
    elif hasattr(model, 'head'):
        model.head = best_classifier
    else:
        raise RuntimeError('Unknown model architecture')

    return model, best_acc


def train_epoch(epoch, model, trainloader, optimizer, criterion, device, warmup_scheduler, beta):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            partial_loss = running_loss / (batch_idx + 1)
            partial_accuracy = 100. * correct / total
            logger.info(f'Epoch {epoch}, Step {batch_idx}, Loss: {partial_loss:.6f}, Accuracy: {partial_accuracy:.2f}%')

        if epoch <= 1 and warmup_scheduler is not None:
            warmup_scheduler.step()

    # Compute final epoch loss and accuracy
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100. * correct / total

    wandb.log({f'epoch_beta{beta:.2f}': epoch, f'train_loss_beta{beta:.2f}': train_loss, f'train_accuracy_beta{beta:.2f}': train_accuracy, f'lr_beta{beta:.2f}': optimizer.param_groups[0]['lr']})
    return train_loss


def test_epoch(epoch, model, testloader, criterion, device, beta):
    """
    Test the model for one epoch.
    Specify epoch=-1 to use for testing after training.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / total
    if epoch != -1:
        logger.info(f'Epoch {epoch}, Val Loss: {test_loss:.6f}, Val Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        wandb.log({f'epoch_beta{beta:.2f}': epoch, f'val_loss_beta{beta:.2f}': test_loss, f'val_accuracy_beta{beta:.2f}': test_accuracy})
    else:
        logger.info(f'Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy

def transfer(model, train_loader, val_loader, beta):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    best_model = None
    best_acc = 0.0

    for epoch in range(1, 31):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler, beta)
        _, val_acc = test_epoch(epoch, model, val_loader, criterion, device, beta)
        if val_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = val_acc
            logger.info(f'New best validation accuracy: {val_acc:.2f} at epoch {epoch}')
        scheduler.step()
    return best_model, best_acc


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

    train_loader, test_loader, val_loader = get_data_loaders(dataset, seed=args.seed)

    criterion = nn.CrossEntropyLoss()

    beta_range = np.arange(0.7, 1.0 - 1e-6, 0.01)

    identifier = f'search_ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(
        project='ct',
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
        # ct_model, val_acc = transfer(ct_model, train_loader, val_loader, beta)
        ct_model, val_acc = linear_probe(ct_model, train_loader, val_loader, beta)

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
