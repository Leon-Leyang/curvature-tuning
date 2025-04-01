"""
This script evaluates the generalization improvement achieved by CT across various image classification datasets.
"""
import torch
import numpy as np
from torch import nn as nn
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from sklearn.linear_model import LogisticRegression
from utils.transfer_learning import FeatureExtractor, WrappedModel, extract_features
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta
from utils.curvature_tuning import replace_module, CT
from train import test_epoch
from loguru import logger
import copy
import argparse
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer_linear_probe(model, pretrained_ds, transfer_ds, reg=1, topk=1):
    """
    Transfer learning.
    """
    logger.debug('Transfer learning with linear probe...')

    # Get the data loaders
    train_loader, _, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}')

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = FeatureExtractor(model, topk)
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    num_classes = DATASET_TO_NUM_CLASSES[transfer_ds]

    # Linear probe
    if topk == 1:
        logistic_regressor = LogisticRegression(max_iter=10000, C=reg)
        logistic_regressor.fit(train_features, train_labels)

        fc = nn.Linear(logistic_regressor.n_features_in_, num_classes).to(device)
        fc.weight.data = torch.tensor(logistic_regressor.coef_, dtype=torch.float).to(device)
        fc.bias.data = torch.tensor(logistic_regressor.intercept_, dtype=torch.float).to(device)
        fc.weight.requires_grad = False
        fc.bias.requires_grad = False
    else:
        wandb.init(project='curvature-tuning', entity='leyang_hu')
        in_features = train_features.shape[1]
        fc = nn.Linear(in_features, num_classes).to(device)
        fc.train()
        train_features = torch.tensor(train_features, dtype=torch.float).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
        train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=True)
        optimizer = torch.optim.Adam(fc.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(30):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = fc(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                wandb.log({'epoch': epoch, 'loss': loss.item()})

        fc.weight.requires_grad = False
        fc.bias.requires_grad = False
        wandb.finish()

    model = WrappedModel(feature_extractor, fc)

    logger.debug('Finishing transfer learning...')
    return model


def replace_and_test_acc(model, beta_vals, dataset, coeff=0.5, model_name=""):
    """
    Replace ReLU with CT and test the model on the specified dataset.
    """
    _, test_loader, _ = get_data_loaders(dataset)

    logger.info(f'Running post-replace accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    _, base_acc = test_epoch(-1, model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)

        # Register the hook for the top-k layer as copy.deepcopy does not copy hooks
        if hasattr(model, 'feature_extractor'):
            new_model.feature_extractor.register_hook(new_model.feature_extractor.topk)

        _, test_acc = test_epoch(-1, new_model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    logger.info(f'Best accuracy for {dataset}: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_metric_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, metric='Accuracy')


def replace_then_lp_test_acc(beta_vals, pretrained_ds, transfer_ds, reg=1, coeff=0.5, topk=1, model_name='resnet18'):
    """
    Replace ReLU with CT and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds, model_name)

    _, test_loader, _ = get_data_loaders(dataset)

    logger.info(f'Running replace then linear probe accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, reg, topk)
    _, base_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, reg, topk)
        _, test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)

    logger.info(
        f'Best accuracy for {dataset}: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_metric_vs_beta(acc_list, beta_list, base_acc, f'{dataset}', metric='Test Accuracy')


def test_acc(dataset, beta_vals, coeff, model_name):
    """
    Test the model's accuracy with different beta values of CT on the same dataset.
    """
    model = get_pretrained_model(dataset, model_name)
    replace_and_test_acc(model, beta_vals, dataset, coeff, model_name)


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on image classification datasets')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--coeff', type=float, default=0.5, help='Coefficient for CT')
    parser.add_argument('--reg', type=float, default=1, help='Regularization strength for Logistic Regression')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topk', type=int, default=1, help='Number of top-k feature layers to use')
    parser.add_argument('--pretrained_ds', type=str, nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='List of transfer datasets')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_coeff{args.coeff}_topk{args.topk}_reg{args.reg}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    betas = np.arange(0.5, 1 - 1e-6, 0.01)

    pretrained_datasets = args.pretrained_ds
    transfer_datasets = args.transfer_ds

    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            fix_seed(args.seed)  # Fix the seed each time

            if pretrained_ds == transfer_ds:  # Test on the same dataset
                # Test generalization
                if result_exists(f'{pretrained_ds}'):
                    logger.info(f'Skipping {pretrained_ds} as result already exists.')
                else:
                    test_acc(pretrained_ds, betas, args.coeff, args.model)

            elif transfer_ds == 'imagenet':  # Skip transfer learning on ImageNet
                continue
            else:  # Test on different datasets
                if result_exists(f'{pretrained_ds}_to_{transfer_ds}'):
                    logger.info(f'Skipping {pretrained_ds} to {transfer_ds} as result already exists.')
                    continue
                replace_then_lp_test_acc(betas, pretrained_ds, transfer_ds, args.reg, args.coeff, args.topk, args.model)


if __name__ == '__main__':
    main()
