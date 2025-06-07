import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image
from utils.utils import MLP, fix_seed
from utils.curvature_tuning import replace_module, CTU


def generate_spiral_data(n_points, noise=0.5, n_turns=3, label_flip=0.05):
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

    n_flips = int(label_flip * n_points)
    flip_indices = np.random.choice(n_points, size=n_flips, replace=False)
    y[flip_indices] = 1 - y[flip_indices]

    return X, y

def plot_decision_boundary(points, target, xx, yy, pred, mesh_dim, color, beta):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["olive" if t == 0 else "palevioletred" for t in target.cpu().numpy()]
    labels = ["Class 1" if t == 0 else "Class 2" for t in target.cpu().numpy()]

    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            points.cpu().numpy()[indices, 0],
            points.cpu().numpy()[indices, 1],
            c=colors[indices[0]],
            alpha=0.6,
            label=label,
            edgecolors="none",
        )

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
    ax.set_title(f"Beta = {beta:.2f}", fontsize=16)
    return fig

def run_experiment(width=20, depth=2, training_steps=2000, beta_vals=None,
                   noise=0.7, n_turns=3, label_flip=0.16, coeff=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    N = 1024
    X, y = generate_spiral_data(n_points=N, noise=noise, n_turns=n_turns, label_flip=label_flip)
    points = torch.from_numpy(X).float().to(device)
    target = torch.from_numpy(y).long().to(device)

    cmap = plt.colormaps["plasma"]
    num_colors = len(beta_vals) + 1
    color_map = [cmap(0.4 * i / (num_colors - 1)) for i in range(num_colors)]

    domain_bound = np.max(np.abs(X)) * 1.2
    mesh_dim = 4000
    xx, yy = np.meshgrid(
        np.linspace(-domain_bound, domain_bound, mesh_dim),
        np.linspace(-domain_bound, domain_bound, mesh_dim),
    )
    grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().to(device)

    losses = []
    frames = []

    base_model = MLP(2, 1, depth, width, nn.ReLU()).to(device)
    optim = torch.optim.AdamW(base_model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)

    for i in range(training_steps):
        output = base_model(points)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

    with torch.no_grad():
        pred = base_model(grid).cpu().numpy()
    fig = plot_decision_boundary(points, target, xx, yy, pred, mesh_dim, color_map[0], beta=1.0)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf))
    losses.append((loss.item(), 1.0))
    print(f"Beta = 1.00, Loss = {loss.item():.4f}")

    for idx, beta in enumerate(beta_vals, start=1):
        model = copy.deepcopy(base_model)
        shared_raw_beta = nn.Parameter(torch.logit(torch.tensor(beta)), requires_grad=False)
        shared_raw_coeff = nn.Parameter(torch.logit(torch.tensor(coeff)), requires_grad=False)
        model = replace_module(model, old_module=nn.ReLU, new_module=CTU,
                               shared_raw_beta=shared_raw_beta, shared_raw_coeff=shared_raw_coeff).to(device)

        with torch.no_grad():
            pred = model(grid).cpu().numpy()
        fig = plot_decision_boundary(points, target, xx, yy, pred, mesh_dim, color_map[idx % len(color_map)], beta=beta)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf))

        output = model(points)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target.float())
        losses.append((loss.item(), beta))
        print(f"Beta = {beta:.2f}, Loss = {loss.item():.4f}")

    # Create GIF from in-memory frames
    gif_path = './figures/demo_classification.gif'
    os.makedirs('./figures', exist_ok=True)
    pause_frames = 25
    extended_frames = frames + [frames[-1]] * pause_frames
    extended_frames[0].save(
        gif_path,
        save_all=True,
        append_images=extended_frames[1:],
        duration=40,
        loop=0
    )

    best_loss, best_beta = min(losses)
    print(f"\nBest beta: {best_beta:.2f} with loss {best_loss:.4f}")

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

    beta_vals = np.arange(0.99, 0.49, -0.01)
    beta_vals = [round(float(b), 2) for b in beta_vals]

    fix_seed(42)
    run_experiment(beta_vals=beta_vals)
