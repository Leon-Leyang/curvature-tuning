import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def plot_val_acc_curve(val_acc_list_all_seeds, beta_range, model_name, transfer_ds, pretrained_ds, save_dir):
    """
    Plot val_acc vs beta with mean and std shading across seeds
    """
    val_acc_array = np.stack(val_acc_list_all_seeds, axis=0)  # shape: (num_seeds, num_beta)
    mean_acc = val_acc_array.mean(axis=0)
    std_acc = val_acc_array.std(axis=0)

    plt.figure(figsize=(10, 7.5))
    plt.plot(beta_range, mean_acc, label='Mean Accuracy', color='black')
    plt.fill_between(beta_range, mean_acc - std_acc, mean_acc + std_acc,
                     alpha=0.3, color='gray', label='±1 std')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Validation Accuracy (%)")
    # plt.title(f"Validation Accuracy vs β\n{model_name} on {transfer_ds}")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'val_acc_curve_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model_name}.pdf')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # set matplotlib fontsize
    plt.rcParams.update({'font.size': 28})  # 30
    plt.rcParams.update({'axes.titlesize': 30})  # 34
    plt.rcParams.update({'axes.labelsize': 28})  # 30
    plt.rcParams.update({'lines.linewidth': 4})  # 8
    plt.rcParams.update({'lines.markersize': 16})  # 24
    plt.rcParams.update({'xtick.labelsize': 24})  # 28
    plt.rcParams.update({'ytick.labelsize': 24})  # 28
    plt.rcParams.update({'xtick.major.size': 16})  # 20
    plt.rcParams.update({'xtick.major.width': 4})  # 4
    plt.rcParams.update({'xtick.minor.size': 10})  # 10
    plt.rcParams.update({'xtick.minor.width': 2})  # 2
    plt.rcParams.update({'ytick.major.size': 16})  # 20
    plt.rcParams.update({'ytick.major.width': 4})  # 4
    plt.rcParams.update({'ytick.minor.size': 10})  # 10
    plt.rcParams.update({'ytick.minor.width': 2})  # 2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 2

    # Disable type3 fonts
    # plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.rcParams.update({'ps.fonttype': 42})

    model_list = ['resnet18', 'resnet50', 'resnet152', 'swin_t', 'swin_s']
    dataset_list = [
        "arabic-characters",
        "arabic-digits",
        "beans",
        "cub200",
        "dtd",
        "fashion-mnist",
        "fgvc-aircraft",
        "flowers102",
        "food101",
        "medmnist/dermamnist",
        "medmnist/octmnist",
        "medmnist/pathmnist",
    ]
    seeds = [42, 43, 44]

    os.makedirs('./val-acc-figures', exist_ok=True)

    for model_name in model_list:
        pretrained_ds = 'imagenette' if 'swin' in model_name else 'imagenet'

        for transfer_ds in dataset_list:
            beta_vals_all_seeds = []
            coeff_vals_all_seeds = []

            val_acc_list_all_seeds = []
            beta_range = np.arange(0.7, 1.0 + 1e-6, 0.01)

            for seed in seeds:
                file_path = f'./results/combined_search_ct_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model_name}_seed{seed}.json'
                if not os.path.exists(file_path):
                    print(f"Warning: Missing file {file_path}")
                    continue
                result = json.load(open(file_path, 'r'))
                val_acc_list = result['val_acc_list']
                val_acc_list_all_seeds.append(val_acc_list)

            if len(val_acc_list_all_seeds) > 0:
                plot_val_acc_curve(val_acc_list_all_seeds, beta_range, model_name, transfer_ds, pretrained_ds,
                                   save_dir='./val-acc-figures')



