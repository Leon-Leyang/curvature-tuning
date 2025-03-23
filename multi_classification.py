"""
This script evaluates the generalization improvement achieved by CT from single-label to multi-label classification.
"""
import torch
import torch.nn as nn
import numpy as np
from utils.data import get_data_loaders
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from utils.transfer_learning import FeatureExtractor, WrappedModel, extract_features
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_metric_vs_beta
from utils.curvature_tuning import replace_module, CT
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
    train_loader, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}')

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = FeatureExtractor(model, topk)
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    num_classes = 40

    # Linear probe
    if topk == 1:
        logistic_regressor = MultiOutputClassifier(LogisticRegression(max_iter=10000, C=reg), n_jobs=-1)
        logistic_regressor.fit(train_features, train_labels)

        # Concatenate each attribute's logistic regression params into a single Linear layer
        W_list, b_list = [], []
        for est in logistic_regressor.estimators_:
            W_list.append(est.coef_)  # shape [1, D]
            b_list.append(est.intercept_)  # shape [1]

        W = np.concatenate(W_list, axis=0)  # shape [40, D]
        b = np.concatenate(b_list, axis=0)  # shape [40]
        fc = nn.Linear(W.shape[1], num_classes).to(device)
        fc.weight.data = torch.from_numpy(W).float().to(device)
        fc.bias.data = torch.from_numpy(b).float().to(device)
        fc.weight.requires_grad = False
        fc.bias.requires_grad = False
    else:
        wandb.init(project='smooth-spline', entity='leyang_hu')
        in_features = train_features.shape[1]
        fc = nn.Linear(in_features, num_classes).to(device)
        fc.train()
        train_features = torch.tensor(train_features, dtype=torch.float).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
        train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=True)
        optimizer = torch.optim.Adam(fc.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

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


def replace_then_lp_test_multi_acc(beta_vals, pretrained_ds, transfer_ds, reg=1, coeff=0.5, topk=1, model_name='resnet18'):
    """
    Replace ReLU with CT and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds, model_name)

    model_name = model.__class__.__name__

    _, test_loader = get_data_loaders(dataset)

    logger.info(f'Running replace then linear probe accuracy test for {model_name} on {dataset}...')

    acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, reg, topk)

    base_acc, acc_by_attr, f1_by_attr, balanced_acc_by_attr, TP, FP, TN, FN = eval_multi_label(transfer_model, test_loader, device)

    # Convert each tensor to a list and format each element
    acc_str = ", ".join([f"{x:.2f}" for x in acc_by_attr.tolist()])
    f1_str = ", ".join([f"{x:.2f}" for x in f1_by_attr.tolist()])
    balanced_acc_str = ", ".join([f"{x:.2f}" for x in balanced_acc_by_attr.tolist()])
    tp_str = ", ".join([str(x) for x in TP.tolist()])
    fp_str = ", ".join([str(x) for x in FP.tolist()])
    tn_str = ", ".join([str(x) for x in TN.tolist()])
    fn_str = ", ".join([str(x) for x in FN.tolist()])

    logger.debug(f'Mean Accuracy: {base_acc:.2f}%')
    logger.debug(f'Per-attribute Accuracy: {acc_str}%')
    logger.debug(f'Per-attribute F1 Score: {f1_str}%')
    logger.debug(f'Per-attribute Balanced Accuracy: {balanced_acc_str}%')
    logger.debug(f'TP: {tp_str}')
    logger.debug(f'FP: {fp_str}')
    logger.debug(f'TN: {tn_str}')
    logger.debug(f'FN: {fn_str}')

    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using CT with beta={beta:.2f}')
        new_model = replace_module(copy.deepcopy(model), nn.ReLU, CT, beta=beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, reg, topk)

        test_acc, acc_by_attr, f1_by_attr, balanced_acc_by_attr, TP, FP, TN, FN = eval_multi_label(transfer_model, test_loader, device)

        # Convert each tensor to a list and format each element
        acc_str = ", ".join([f"{x:.2f}" for x in acc_by_attr.tolist()])
        f1_str = ", ".join([f"{x:.2f}" for x in f1_by_attr.tolist()])
        balanced_acc_str = ", ".join([f"{x:.2f}" for x in balanced_acc_by_attr.tolist()])
        tp_str = ", ".join([str(x) for x in TP.tolist()])
        fp_str = ", ".join([str(x) for x in FP.tolist()])
        tn_str = ", ".join([str(x) for x in TN.tolist()])
        fn_str = ", ".join([str(x) for x in FN.tolist()])

        logger.debug(f'Mean Accuracy: {test_acc:.2f}%')
        logger.debug(f'Per-attribute Accuracy: {acc_str}%')
        logger.debug(f'Per-attribute F1 Score: {f1_str}%')
        logger.debug(f'Per-attribute Balanced Accuracy: {balanced_acc_str}%')
        logger.debug(f'TP: {tp_str}')
        logger.debug(f'FP: {fp_str}')
        logger.debug(f'TN: {tn_str}')
        logger.debug(f'FN: {fn_str}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)

    logger.info(
        f'Best accuracy for {dataset}: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_metric_vs_beta(acc_list, beta_list, base_acc, f'{dataset}', model_name, metric='Mean Accuracy')


def eval_multi_label(model, testloader, device='cuda'):
    """
    Evaluates a multi-label classification model.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            # Handle special case (e.g., CelebA) where targets come as a list of tensors.
            if isinstance(targets, list) and isinstance(targets[0], torch.Tensor):
                targets = torch.stack(targets, dim=0)  # shape: (num_attributes, batch_size)
                targets = targets.T  # shape: (batch_size, num_attributes)
                targets = targets.float()

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu()

            # Convert ground-truth from -1/+1 to 0/1.
            targets_01 = (targets + 1) // 2

            all_preds.append(probs)
            all_targets.append(targets_01)

    # Concatenate all batches.
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N, num_attributes)
    all_targets = torch.cat(all_targets, dim=0)  # shape: (N, num_attributes)

    # Threshold probabilities at 0.5 to obtain binary predictions {0,1}.
    pred_labels = (all_preds >= 0.5).float()

    # Compute per-attribute accuracy.
    correct_by_attr = (pred_labels == all_targets).float().sum(dim=0)
    total_samples = all_targets.size(0)
    acc_by_attr = correct_by_attr / total_samples

    # Mean accuracy across attributes.
    mA = acc_by_attr.mean().item()

    # Compute per-attribute F1 scores.
    TP = ((pred_labels == 1) & (all_targets == 1)).float().sum(dim=0)
    FP = ((pred_labels == 1) & (all_targets == 0)).float().sum(dim=0)
    TN = ((pred_labels == 0) & (all_targets == 0)).float().sum(dim=0)
    FN = ((pred_labels == 0) & (all_targets == 1)).float().sum(dim=0)
    f1_by_attr = 2 * TP / (2 * TP + FP + FN + 1e-8)

    # Compute per-attribute balanced accuracy.
    TN = ((pred_labels == 0) & (all_targets == 0)).float().sum(dim=0)
    TPR = TP / (TP + FN + 1e-8)  # Sensitivity or Recall for positives.
    TNR = TN / (TN + FP + 1e-8)  # Specificity for negatives.
    balanced_acc_by_attr = (TPR + TNR) / 2

    return (mA * 100.0,
            acc_by_attr * 100.0,
            f1_by_attr * 100.0,
            balanced_acc_by_attr * 100.0,
            TP, FP, TN, FN)


def get_args():
    parser = argparse.ArgumentParser(description='Generalization experiments on multi-label image classification datasets')
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
                        default=['imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['celeb_a'], help='List of transfer datasets')
    return parser.parse_args()


def main():
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(
        name=f'{f_name}_coeff{args.coeff}_topk{args.topk}_reg{args.reg}_{args.model}_more_ds_seed{args.seed}')
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
            replace_then_lp_test_multi_acc(betas, pretrained_ds, transfer_ds, args.reg, args.coeff, args.topk, args.model)


if __name__ == '__main__':
    main()