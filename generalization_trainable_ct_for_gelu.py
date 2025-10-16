"""
This script evaluates the generalization improvement achieved by Trainable CT across various image classification datasets,
now with LR cross-validation:
- Trainable CT: cross-validate ct_lr over {1e-1, 1e-2, 1e-3, 1e-4} (base lr kept at 1e-3)
- LoRA: cross-validate lr over {1e-3, 1e-4, 1e-5}

For each method, the LR that yields the best validation accuracy is used to evaluate on the test set. The script logs
per-LR validation accuracies to Weights & Biases using separate runs, records which LR won, and saves `best_lr` in
the results JSON.
"""
import torch
from torch import nn as nn
from torch import optim
from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, get_file_name, fix_seed, set_logger, save_result_json
from utils.curvature_tuning import TrainableCTU, replace_module_dynamic, get_mean_beta_and_coeff
from utils.lora import get_lora_model
from train import train_epoch, test_epoch, WarmUpLR
from loguru import logger
import copy
import argparse
import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def transfer(model, train_loader, val_loader, lr=1e-3, ct_lr=1e-1):
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
        {'params': ct_params, 'lr': ct_lr},
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
    parser.add_argument('--transfer_train_bs', type=int, default=32, help='Batch size for transfer learning')
    parser.add_argument('--transfer_test_bs', type=int, default=800, help='Batch size for transfer learning test')
    return parser.parse_args()


def main():
    args = get_args()

    lora_rank = 1

    transfer_ds_alias = args.transfer_ds.replace('/', '-')

    result_path = {
        'train_ct': f'./results/train_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json',
        'lora': f'./results/lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.json'
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

    train_loader, test_loader, val_loader = get_data_loaders(
        dataset,
        seed=args.seed,
        train_batch_size=args.transfer_train_bs,
        test_batch_size=args.transfer_test_bs
    )

    criterion = nn.CrossEntropyLoss()

    # ============================
    # Trainable CT: LR CV over ct_lr
    # ============================
    logger.info('Testing Trainable CT with ct_lr cross-validation...')
    dummy_input_shape = (1, 3, 224, 224)
    # Base CT model (untrained) — we'll deepcopy this per ct_lr
    base_ct_model = replace_module_dynamic(
        copy.deepcopy(model),
        dummy_input_shape,
        old_module=nn.GELU,
        new_module=TrainableCTU,
        raw_beta=0.5767,
        raw_coeff=10.0
    ).to(device)

    num_params_ct = sum(p.numel() for p in base_ct_model.parameters() if p.requires_grad)
    logger.info(f'[CT] Number of trainable parameters: {num_params_ct}')

    # Probe initial (pre-train) beta/coeff for info
    init_beta, init_coeff = get_mean_beta_and_coeff(base_ct_model)
    logger.info(f'[CT] Initial Mean Beta: {init_beta:.6f}, Initial Mean Coeff: {init_coeff:.6f}')

    ct_lr_candidates = [1e-1, 1e-2, 1e-3, 1e-4]
    ct_lr_to_val = {}
    ct_lr_to_model = {}

    for ct_lr in ct_lr_candidates:
        run_name = f'train_ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}_ctlr{ct_lr:.0e}'
        wandb.init(
            project='ct-new',
            name=run_name,
            config={**vars(args), 'ct_lr': ct_lr, 'base_lr': 1e-3},
            reinit=True,
        )
        logger.info(f'[CT] Starting transfer with ct_lr={ct_lr:.0e}, base lr=1e-3...')
        m = copy.deepcopy(base_ct_model).to(device)
        m, val_acc = transfer(m, train_loader, val_loader, lr=1e-3, ct_lr=ct_lr)
        ct_lr_to_val[ct_lr] = float(val_acc)
        ct_lr_to_model[ct_lr] = m
        wandb.log({'val_accuracy': val_acc, 'num_params': num_params_ct})
        wandb.finish()
        logger.info(f'[CT] Validation Accuracy (ct_lr={ct_lr:.0e}): {val_acc:.2f}%')

    # Pick the best ct_lr by validation accuracy
    ct_best_lr = max(ct_lr_to_val, key=ct_lr_to_val.get)
    ct_best_val = ct_lr_to_val[ct_best_lr]
    ct_model = ct_lr_to_model[ct_best_lr]
    logger.info(f'[CT] Best ct_lr={ct_best_lr:.0e} with val_acc={ct_best_val:.2f}%')

    # Test the best CT model
    logger.info('[CT] Evaluating best CT model on test set...')
    _, ct_acc = test_epoch(-1, ct_model, test_loader, criterion, device)
    logger.info(f'[CT] Test Accuracy (best ct_lr): {ct_acc:.2f}%')

    # Recompute mean beta/coeff post-training on the selected model
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'[CT] Final Mean Beta: {mean_beta:.6f}, Final Mean Coeff: {mean_coeff:.6f}')

    # Summary W&B run for CT
    ct_summary_name = f'train_ct_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(project='ct-new', name=ct_summary_name, config=vars(args), reinit=True)
    wandb.log({
        'test_accuracy': ct_acc,
        'best_lr': ct_best_lr,
        'best_val_accuracy': ct_best_val,
        'num_params': num_params_ct,
        **{f'val_accuracy_ct_lr_{lr:.0e}': acc for lr, acc in ct_lr_to_val.items()}
    })
    wandb.finish()

    # Save the CT model checkpoint
    os.makedirs('./ckpts', exist_ok=True)
    ct_ckpt_path = f'./ckpts/train_ct_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth'
    torch.save(ct_model.state_dict(), ct_ckpt_path)
    logger.info(f'[CT] Model saved to {ct_ckpt_path}')

    # ===========
    # LoRA: LR CV
    # ===========
    logger.info('Testing LoRA with lr cross-validation...')
    lora_alpha = lora_rank

    # Base LoRA model template; we'll deepcopy/refresh the LoRA model per lr
    base_lora_model = get_lora_model(copy.deepcopy(model), r=lora_rank, alpha=lora_alpha).to(device)
    # Replace the last layer with standard linear (ensuring correct head)
    if 'swin' not in args.model:
        base_lora_model.fc = nn.Linear(
            in_features=base_lora_model.fc.in_features,
            out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]
        ).to(device)
    else:
        base_lora_model.head = nn.Linear(
            in_features=base_lora_model.head.in_features,
            out_features=DATASET_TO_NUM_CLASSES[args.transfer_ds]
        ).to(device)

    num_params_lora = sum(p.numel() for p in base_lora_model.parameters() if p.requires_grad)
    logger.info(f'[LoRA] Number of trainable parameters: {num_params_lora}')

    lora_lr_candidates = [1e-3, 1e-4, 1e-5]
    lora_lr_to_val = {}
    lora_lr_to_model = {}

    for lr in lora_lr_candidates:
        run_name = f'lora_rank{lora_rank}_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}_lr{lr:.0e}'
        wandb.init(
            project='ct-new',
            name=run_name,
            config={**vars(args), 'lr': lr, 'lora_rank': lora_rank, 'lora_alpha': lora_alpha},
            reinit=True,
        )
        logger.info(f'[LoRA] Starting transfer with lr={lr:.0e}...')
        # Fresh LoRA model for each LR
        m = copy.deepcopy(base_lora_model).to(device)

        m, val_acc = transfer(m, train_loader, val_loader, lr=lr)  # ct_lr not relevant for plain LoRA
        lora_lr_to_val[lr] = float(val_acc)
        lora_lr_to_model[lr] = m
        wandb.log({'val_accuracy': val_acc, 'num_params': num_params_lora})
        wandb.finish()
        logger.info(f'[LoRA] Validation Accuracy (lr={lr:.0e}): {val_acc:.2f}%')

    # Pick the best LoRA lr by validation accuracy
    lora_best_lr = max(lora_lr_to_val, key=lora_lr_to_val.get)
    lora_best_val = lora_lr_to_val[lora_best_lr]
    lora_model = lora_lr_to_model[lora_best_lr]
    logger.info(f'[LoRA] Best lr={lora_best_lr:.0e} with val_acc={lora_best_val:.2f}%')

    # Test the best LoRA model
    logger.info('[LoRA] Evaluating best LoRA model on test set...')
    _, lora_acc = test_epoch(-1, lora_model, test_loader, criterion, device)
    logger.info(f'[LoRA] Test Accuracy (best lr): {lora_acc:.2f}%')

    # Summary W&B run for LoRA
    lora_summary_name = f'lora_rank{lora_rank}_{args.pretrained_ds}_to_{args.transfer_ds}_{args.model}_seed{args.seed}'
    wandb.init(project='ct-new', name=lora_summary_name, config=vars(args), reinit=True)
    wandb.log({
        'test_accuracy': lora_acc,
        'best_lr': lora_best_lr,
        'best_val_accuracy': lora_best_val,
        'num_params': num_params_lora,
        **{f'val_accuracy_lr_{lr:.0e}': acc for lr, acc in lora_lr_to_val.items()}
    })
    wandb.finish()

    # Save the LoRA model checkpoint
    lora_ckpt_path = f'./ckpts/lora_rank{lora_rank}_{args.pretrained_ds}_to_{transfer_ds_alias}_{args.model}_seed{args.seed}.pth'
    torch.save(lora_model.state_dict(), lora_ckpt_path)
    logger.info(f'[LoRA] Model saved to {lora_ckpt_path}')

    # Log the summary
    logger.info(f'Trainable CT model trainable parameters: {num_params_ct}')
    logger.info(f'LoRA model trainable parameters: {num_params_lora}')
    logger.info(f'Trainable CT params/LoRA params: {num_params_ct / num_params_lora:.2f}')
    rel_improve_lora = (ct_acc - lora_acc) / lora_acc
    logger.info(f'Trainable CT Accuracy: {ct_acc:.2f}%')
    logger.info(f'LoRA Accuracy: {lora_acc:.2f}%')
    logger.info(f'Relative accuracy improvement over LoRA: {rel_improve_lora * 100:.2f}%')
    mean_beta, mean_coeff = get_mean_beta_and_coeff(ct_model)
    logger.info(f'Mean Beta: {mean_beta:.6f}, Mean Coeff: {mean_coeff:.6f}')

    # Save the results
    os.makedirs('./results', exist_ok=True)

    save_result_json(
        result_path['train_ct'],
        num_params_ct, ct_acc, beta=mean_beta, coeff=mean_coeff, best_ct_lr=ct_best_lr)
    save_result_json(
        result_path['lora'],
        num_params_lora, lora_acc, best_lr=lora_best_lr)
    logger.info('Results saved to ./results/')


if __name__ == '__main__':
    main()
