"""
This file shows the smoothing of the classification model's decision boundary by applying CT with different beta values.
"""
import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import MLP, get_file_name, fix_seed, set_logger, get_log_file_path
from utils.curvature_tuning import replace_module, CT
from loguru import logger


def generate_spiral_data(n_points, noise=0.5, n_turns=3, label_flip=0.05):
    """
    Generate a spiral dataset with more turns, with reduced noise for smaller radii.

    Args:
    - n_points (int): Total number of points in the dataset.
    - noise (float): Base amount of noise to add to the points.
    - n_turns (int): Number of turns in the spiral.

    Returns:
    - X (numpy.ndarray): The coordinates of the points (n_points x 2).
    - y (numpy.ndarray): The class labels for the points (0 or 1).
    """
    n = n_points // 2
    theta = np.linspace(np.pi / 2, n_turns * np.pi, n)
    r_a = theta
    x_a = r_a * np.cos(theta) + noise * np.random.randn(n)
    y_a = r_a * np.sin(theta) + noise * np.random.randn(n)
    r_b = theta
    x_b = r_b * np.cos(theta + np.pi) + noise * np.random.randn(n)
    y_b = r_b * np.sin(theta + np.pi) + noise * np.random.randn(n)

    X_a = np.vstack((x_a, y_a)).T
    X_b = np.vstack((x_b, y_b)).T
    X = np.vstack((X_a, X_b))
    y = np.hstack((np.zeros(n), np.ones(n)))

    # Flip a portion of the labels based on label_flip
    n_flips = int(label_flip * n_points)
    flip_indices = np.random.choice(n_points, size=n_flips, replace=False)
    y[flip_indices] = 1 - y[flip_indices]

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

    ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])


# Use the helper function in the plotting loop
def plot_classification_bond(
    width: int, depth: int, training_steps=2000, beta_vals=[0.9], noise=0.4, n_turns=3, label_flip=0.05, c=0.5, colors=None
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

    N = 1024  # Number of training points

    # Data generation
    logger.debug("Generating data...")
    X, y = generate_spiral_data(n_points=N, noise=noise, n_turns=n_turns, label_flip=label_flip)
    points = torch.from_numpy(X).float().to(device)
    target = torch.from_numpy(y).long().to(device)

    # Model and optimizer definition
    relu_model = MLP(2, 1, depth, width, nn.ReLU()).to(device)
    optim = torch.optim.AdamW(relu_model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)

    # Training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = relu_model(points)[:, 0]
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

    domain_bound = np.max(np.abs(X)) * 1.2  # Extend slightly beyond data range
    mesh_dim = 4000
    with torch.no_grad():
        xx, yy = np.meshgrid(
            np.linspace(-domain_bound, domain_bound, mesh_dim),
            np.linspace(-domain_bound, domain_bound, mesh_dim),
        )
        grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().to(device)

    # Row configurations for each activation type
    activation_configs = [
        ("CT", c),
    ]

    # Create subplots
    num_cols = len(beta_vals) + 1  # Include one column for the ReLU baseline
    num_rows = len(activation_configs)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for row, (name, coeff) in enumerate(activation_configs):
        with torch.no_grad():
            pred = relu_model(grid).cpu().numpy()
        plot_decision_boundary(
            axs[row, 0], points, target, xx, yy, pred, None, mesh_dim, colors[0]
        )

        # Plot for each beta
        for col, beta in enumerate(beta_vals, start=1):
            new_model = replace_module(copy.deepcopy(relu_model), nn.ReLU, CT, beta=beta, coeff=coeff)
            with torch.no_grad():
                pred = new_model(grid).cpu().numpy()
            plot_decision_boundary(
                axs[row, col],
                points,
                target,
                xx,
                yy,
                pred,
                None,
                mesh_dim,
                colors[col],
            )

    # Adjust layout and save the figure
    plt.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(f'./figures/{get_file_name(get_log_file_path())}_classification.svg')
    plt.show()


if __name__ == "__main__":
    beta_vals = [0.9, 0.5]
    cmap = plt.colormaps["Dark2"]
    colors = [cmap(2), cmap(1), cmap(0)]
    width = 20
    depth = 2
    training_steps = 2000
    noise = 0.7
    n_turns = 3
    coeff = 0.5
    label_flip = 0.16

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{width}_depth{depth}_steps{training_steps}_noise{noise}_turns{n_turns}_coeff{coeff}_seed42')
    fix_seed(42)
    plot_classification_bond(width, depth, training_steps, beta_vals, noise, n_turns, label_flip, coeff, colors=colors)
