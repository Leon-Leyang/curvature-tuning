"""
This file generates the plots for the toy example.
"""
import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import MLP, get_file_name, fix_seed, set_logger, get_log_file_path
from utils.curvature_tuning import replace_module_dynamic, TrainableCTU
from utils.lora import get_lora_model
from loguru import logger
import numpy as np


def train(model, points, target, training_steps):
    optim = torch.optim.AdamW(model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)
    # Training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = model(points)[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, target.float()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 100 == 0:
                pbar.update(1)
                pbar.set_description(f"Loss {loss.item()}")
    return loss.item()


def generate_circular_data(n_points):
    """
    Generate a concentric circle dataset.

    Args:
    - n_points (int): Total number of points in the dataset.
    Returns:
    - X (numpy.ndarray): The coordinates of the points (n_points x 2).
    - y (numpy.ndarray): The class labels for the points (0 or 1).
    """
    n_points_small_circle = int(n_points / 4)
    n_points_large_circle = n_points - n_points_small_circle

    # First circle (inner, class 0)
    radius_inner = 1.0
    angles_inner = np.linspace(0, 2 * np.pi, n_points_small_circle, endpoint=False)
    x_inner = radius_inner * np.cos(angles_inner)
    y_inner = radius_inner * np.sin(angles_inner)
    labels_inner = np.zeros(n_points_small_circle, dtype=int)

    # Second circle (outer, class 1)
    radius_outer = 3
    angles_outer = np.linspace(0, 2 * np.pi, n_points_large_circle, endpoint=False)
    x_outer = radius_outer * np.cos(angles_outer)
    y_outer = radius_outer * np.sin(angles_outer)
    labels_outer = np.ones(n_points_large_circle, dtype=int)

    # Combine the data
    X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
    y = np.concatenate((labels_inner, labels_outer))

    return X, y


def plot_decision_boundary(ax, points, target, xx, yy, pred, title, mesh_dim, color='red'):
    """
    Plot decision boundary and data points on a single axis.

    Args:
    - ax: The matplotlib axis to plot on.
    - points: Torch tensor of data points.
    - target: Torch tensor of target labels.
    - xx, yy: Meshgrid arrays for plotting the decision boundary.
    - pred: Predictions from the model for the meshgrid.
    - title: Title for the subplot.
    - mesh_dim: Dimension of the meshgrid.
    """
    colors = ["olive" if t == 0 else "palevioletred" for t in target.cpu().numpy()]
    labels = ["Class 1" if t == 0 else "Class 2" for t in target.cpu().numpy()]

    # Plot data points with unique labels for the legend
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            points.cpu().numpy()[indices, 0],
            points.cpu().numpy()[indices, 1],
            c=colors[indices[0]],  # Same color for the label
            alpha=0.6,
            label=label,
            edgecolors="none",
        )

    # Plot decision boundary
    ax.contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=[color],
        linewidths=[5],
    )

    ax.set_xticks([])
    ax.set_yticks([])


# Use the helper function in the plotting loop
def plot_classification_bond(
    width: int, depth: int, training_steps=2000, finetune_step=4000, colors=None, init_beta=0.5
) -> None:
    """
    Plot the decision boundary of the model for CT.

    Args:
    - width (int): Width of the MLP.
    - depth (int): Depth of the MLP.
    - training_steps (int): Number of training steps.
    - beta_vals (list of float): List of beta values to test.
    - noise (float): Noise level for spiral data.
    - n_turns (int): Number of spiral turns.
    - c (float): Coefficient for CT.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    N = 180  # Number of training points

    # Data generation
    logger.debug("Generating data...")
    X, y = generate_circular_data(n_points=N)
    points = torch.from_numpy(X).float().to(device)
    target = torch.from_numpy(y).long().to(device)

    # Model and optimizer definition
    baseline_model = MLP(2, 1, depth, width, nn.ReLU()).to(device)
    base_loss = train(baseline_model, points, target, training_steps)

    domain_bound = np.max(np.abs(X)) * 1.2  # Extend slightly beyond data range
    mesh_dim = 4000
    with torch.no_grad():
        xx, yy = np.meshgrid(
            np.linspace(-domain_bound, domain_bound, mesh_dim),
            np.linspace(-domain_bound, domain_bound, mesh_dim),
        )
        grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().to(device)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(5 * 3, 5))
    axs = np.expand_dims(axs, axis=0)

    with torch.no_grad():
        pred = baseline_model(grid).cpu().numpy()
    plot_decision_boundary(
        axs[0, 0], points, target, xx, yy, pred, None, mesh_dim, colors[0]
    )

    # Plot for LoRA
    lora_model = get_lora_model(copy.deepcopy(baseline_model), r=1, alpha=1).to(device)
    lora_loss = train(lora_model, points, target, training_steps=finetune_step)
    with torch.no_grad():
        pred = lora_model(grid).cpu().numpy()
    plot_decision_boundary(
        axs[0, 1],
        points,
        target,
        xx,
        yy,
        pred,
        None,
        mesh_dim,
        colors[1],
    )

    # Plot for CT
    dummy_input_shape = (1, 2)
    ct_model = replace_module_dynamic(copy.deepcopy(baseline_model), dummy_input_shape, old_module=nn.ReLU,
                                      new_module=TrainableCTU, raw_beta=torch.logit(torch.tensor(init_beta)).item()).to(device)
    ct_loss = train(ct_model, points, target, training_steps=finetune_step)

    if ct_loss > lora_loss or lora_loss > base_loss:
        return

    with torch.no_grad():
        pred = ct_model(grid).cpu().numpy()
    plot_decision_boundary(
        axs[0, 2],
        points,
        target,
        xx,
        yy,
        pred,
        None,
        mesh_dim,
        colors[2],
    )

    # Adjust layout and save the figure
    plt.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'./figures/{get_file_name(get_log_file_path())}.pdf')
    plt.show()


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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--training_steps", type=int, default=4000)
    parser.add_argument("--finetune_step", type=int, default=4000)
    parser.add_argument("--init_beta", type=float, default=0.4)

    args = parser.parse_args()

    cmap = plt.colormaps["Dark2"]
    colors = [cmap(2), cmap(1), cmap(0)]
    width = args.width
    depth = args.depth
    training_steps = args.training_steps
    finetune_step = args.finetune_step
    init_beta = args.init_beta

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{width}_depth{depth}_beta{init_beta:.2f}')
    fix_seed(42)
    plot_classification_bond(width, depth, training_steps, finetune_step, colors=colors, init_beta=init_beta)
