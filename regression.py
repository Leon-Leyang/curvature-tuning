"""
This script evaluates the generalization improvement achieved by CT from classification to regression.
"""
import torch
import torch.nn as nn
import numpy as np
from utils.data import get_data_loaders
from sklearn.linear_model import LinearRegression
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta
from utils.curvature_tuning import replace_module, CT
from utils.transfer_learning import FeatureExtractor, WrappedModel, extract_features
from loguru import logger
import copy
import argparse
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer_linear_probe(model, pretrained_ds, transfer_ds, topk=1):
    """
    Transfer learning.
    """
    logger.debug('Transfer learning with linear probe...')

    # Get the data loaders
    if transfer_ds == 'dsprites':
        train_loader, _, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}', train_size=50000, test_size=10000)
    else:
        train_loader, _, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}')

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = FeatureExtractor(model, topk)
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    num_classes = 1

    # Linear probe
    if topk == 1:
        linear_regressor = LinearRegression()
        linear_regressor.fit(train_features, train_labels)

        fc = nn.Linear(linear_regressor.coef_.shape[0], 1).to(device)
        fc.weight.data = torch.tensor(linear_regressor.coef_, dtype=torch.float).unsqueeze(0).to(device)
        fc.bias.data = torch.tensor(linear_regressor.intercept_, dtype=torch.float).view(1).to(device)
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
        criterion = nn.MSELoss()

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


def replace_then_lp_test_mse(beta_vals, pretrained_ds, transfer_ds, coeff=0.5, topk=1, model_name='resnet18'):
    """
    Replace ReLU with CT and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds, model_name)

    model_name = model.__class__.__name__

    if transfer_ds == 'dsprites':
        _, test_loader, _ = get_data_loaders(dataset, train_size=50000, test_size=10000)
    else:
        _, test_loader, _ = get_data_loaders(dataset)

    logger.info(f'Running replace then linear probe mse test for {model_name} on {dataset}...')

    mse_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, topk)
    base_mse = eval_mse(transfer_model, test_loader, device)
    logger.debug(f'Mean Squared Error: {base_mse:.4f}')
    best_mse = base_mse
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, topk)
        test_mse = eval_mse(transfer_model, test_loader, device)
        logger.debug(f'Mean Squared Error: {test_mse:.4f}')
        if test_mse < best_mse:
            best_mse = test_mse
            best_beta = beta

        mse_list.append(test_mse)
        beta_list.append(beta)

    mse_list.append(base_mse)
    beta_list.append(1)
    logger.info(
        f'Best MSE for {dataset}: {best_mse:.4f} with beta={best_beta:.2f}, compared to ReLU MSE: {base_mse:.4f}')

    plot_metric_vs_beta(mse_list, beta_list, base_mse, f'{dataset}', model_name, metric='MSE')


def eval_mse(model, testloader, device='cuda'):
    """
    Computes Mean Squared Error (MSE) for regression tasks.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()  # Ensure targets are floats
            preds = model(inputs).cpu()  # Get model predictions

            all_preds.append(preds)
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N,)
    all_targets = torch.cat(all_targets, dim=0)  # shape: (N,)

    # Compute Mean Squared Error (MSE)
    mse = torch.mean((all_preds - all_targets) ** 2).item()
    return mse


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on regression datasets')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to test'
    )
    parser.add_argument('--coeff', type=float, default=0.5, help='Coefficient for CT')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topk', type=int, default=1, help='Number of top-k feature layers to use')
    parser.add_argument('--pretrained_ds', type=str, nargs='+',
                        default=['imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['dsprites'], help='List of transfer datasets')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_coeff{args.coeff}_topk{args.topk}_{args.model}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    betas = np.arange(0.5, 1 - 1e-6, 0.01)

    pretrained_datasets = args.pretrained_ds
    transfer_datasets = args.transfer_ds

    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            fix_seed(args.seed)  # Fix the seed each time

            if result_exists(f'{pretrained_ds}_to_{transfer_ds}'):
                logger.info(f'Skipping {pretrained_ds} to {transfer_ds} as result already exists.')
                continue
            replace_then_lp_test_mse(betas, pretrained_ds, transfer_ds, args.coeff, args.topk, args.model)


if __name__ == '__main__':
    main()
