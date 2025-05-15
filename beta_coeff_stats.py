import torch
import torch.nn as nn
import copy
import os
from utils.utils import get_pretrained_model
from utils.data import DATASET_TO_NUM_CLASSES
from utils.curvature_tuning import replace_module_dynamic, TrainableCTU
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_beta_and_coeff(model):
    """
    Iterate through the model to get the list of beta and coeff values.
    """
    beta_vals = []
    coeff_vals = []

    for module in model.modules():
        if isinstance(module, TrainableCTU):
            beta = module.beta.detach().flatten()
            coeff = module.coeff.detach().flatten()
            assert 0 <= beta.min() <= beta.max() <= 1
            assert 0 <= coeff.min() <= coeff.max() <= 1
            beta_vals.append(beta)
            coeff_vals.append(coeff)

    return beta_vals, coeff_vals


def plot_mean_std_distribution(all_vals, varname, model_name, transfer_ds, pretrained_ds, save_dir):
    """
    all_vals: list of 1D numpy arrays (one per seed)
    varname: str, either 'beta' or 'coeff'
    """
    # Normalize and compute histogram for each seed
    bin_edges = np.linspace(0, 1, 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    all_densities = []

    for vals in all_vals:
        hist, _ = np.histogram(vals, bins=bin_edges, density=True)
        all_densities.append(hist)

    all_densities = np.stack(all_densities, axis=0)  # shape: (num_seeds, num_bins)
    mean_density = all_densities.mean(axis=0)
    std_density = all_densities.std(axis=0)

    # Choose color
    color = 'blue' if varname == 'beta' else 'green'

    # Plot with shaded region
    plt.figure(figsize=(10, 7.5))
    plt.plot(bin_centers, mean_density, label='Mean Density', color=color)
    plt.fill_between(bin_centers,
                     mean_density - std_density,
                     mean_density + std_density,
                     alpha=0.3,
                     color=color,
                     label='±1 std')
    plt.xlabel(r"$\beta$" if varname == "beta" else r"$c$")
    plt.ylabel("Density")
    # plt.title(f"{varname.capitalize()} Distribution\n{model_name} on {transfer_ds}")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{varname}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model_name}.pdf')
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

    os.makedirs('./stats-figures', exist_ok=True)

    count = 0
    for model_name in model_list:
        pretrained_ds = 'imagenette' if 'swin' in model_name else 'imagenet'
        model = get_pretrained_model(pretrained_ds, model_name)

        for transfer_ds in dataset_list:
            beta_vals_all_seeds = []
            coeff_vals_all_seeds = []

            if 'swin' not in model_name:
                model.fc = nn.Linear(in_features=model.fc.in_features,
                                     out_features=DATASET_TO_NUM_CLASSES[transfer_ds]).to(device)
            else:
                model.head = nn.Linear(in_features=model.head.in_features,
                                       out_features=DATASET_TO_NUM_CLASSES[transfer_ds]).to(device)

            dummy_input_shape = (1, 3, 224, 224)
            ct_model = replace_module_dynamic(copy.deepcopy(model), dummy_input_shape, old_module=nn.ReLU,
                                              new_module=TrainableCTU).to(device)

            for seed in seeds:
                file_path = f'./ckpts/ct_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model_name}_seed{seed}.pth'

                ct_model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))

                beta_vals, coeff_vals = get_beta_and_coeff(ct_model)

                beta_vals_all_seeds.append(torch.cat(beta_vals).cpu().numpy())
                coeff_vals_all_seeds.append(torch.cat(coeff_vals).cpu().numpy())

            # Print mean and std
            beta_all = np.concatenate(beta_vals_all_seeds)
            coeff_all = np.concatenate(coeff_vals_all_seeds)
            beta_mean, beta_std = beta_all.mean(), beta_all.std()
            coeff_mean, coeff_std = coeff_all.mean(), coeff_all.std()

            print(f"[{model_name} | {transfer_ds}]")
            print(f"  β   : mean = {beta_mean:.2f}, std = {beta_std:.2f}")
            print(f"  c   : mean = {coeff_mean:.2f}, std = {coeff_std:.2f}")

            plot_mean_std_distribution(beta_vals_all_seeds, "beta", model_name, transfer_ds, pretrained_ds,
                                       save_dir='./stats-figures')
            plot_mean_std_distribution(coeff_vals_all_seeds, "coeff", model_name, transfer_ds, pretrained_ds,
                                       save_dir='./stats-figures')

