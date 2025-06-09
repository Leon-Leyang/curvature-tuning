import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from utils.utils import MLP, get_file_name, fix_seed, set_logger
from utils.curvature_tuning import replace_module, CTU


def generate_curve_data(n_points, noise=0.1):
    X = np.linspace(-np.pi / 2, 3 * np.pi / 2, n_points)
    y = 1.2 * np.sin(2 * X) + noise * np.random.randn(n_points)
    return X, y

def plot_regression_curve(X, y, x_range, true_curve, prediction, color, beta):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X, y, label="Training Data", color="blue")
    ax.plot(x_range, true_curve, label="True Curve", color="black", linewidth=2.5)
    ax.plot(x_range, prediction, label="Prediction", color=color, linewidth=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Beta = {beta:.3f}", fontsize=30)
    return fig

def run_regression_experiment(width=64, depth=8, training_steps=20000, beta_vals=None,
                               noise=0.3, coeff=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    N = 30
    X, y = generate_curve_data(n_points=N, noise=noise)
    points = torch.from_numpy(X).float().to(device).unsqueeze(-1)
    target = torch.from_numpy(y).float().to(device).unsqueeze(-1)

    cmap = plt.colormaps["plasma"]
    num_colors = len(beta_vals) + 1
    color_map = [cmap(0.4 * i / (num_colors - 1)) for i in range(num_colors)]

    x_range = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    x_range_torch = torch.from_numpy(x_range).float().to(device)
    true_curve = 1.2 * np.sin(2 * x_range)

    losses = []
    frames = []

    base_model = MLP(1, 1, depth, width, nn.ReLU()).to(device)
    optim = torch.optim.AdamW(base_model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)
    for _ in range(training_steps):
        output = base_model(points)
        loss = nn.MSELoss()(output, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

    with torch.no_grad():
        pred = base_model(x_range_torch).squeeze().cpu().numpy()
    fig = plot_regression_curve(X, y, x_range, true_curve, pred, color_map[0], beta=1.0)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf))
    output = base_model(points)
    loss = nn.MSELoss()(output, target)
    losses.append((loss.item(), 1.0))
    print(f"Beta = 1.000, Loss = {loss.item():.4f}")

    for idx, beta in enumerate(beta_vals, start=1):
        model = copy.deepcopy(base_model)
        shared_raw_beta = nn.Parameter(torch.logit(torch.tensor(beta)), requires_grad=False)
        shared_raw_coeff = nn.Parameter(torch.logit(torch.tensor(coeff)), requires_grad=False)
        model = replace_module(model, old_module=nn.ReLU, new_module=CTU,
                               shared_raw_beta=shared_raw_beta, shared_raw_coeff=shared_raw_coeff).to(device)
        with torch.no_grad():
            pred = model(x_range_torch).squeeze().cpu().numpy()
        fig = plot_regression_curve(X, y, x_range, true_curve, pred, color_map[idx % len(color_map)], beta=beta)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf))

        output = model(points)
        loss = nn.MSELoss()(output, target)
        losses.append((loss.item(), beta))
        print(f"Beta = {beta:.3f}, Loss = {loss.item():.4f}")

    os.makedirs('./figures', exist_ok=True)
    gif_path = './figures/demo_regression.gif'
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
    print(f"\nBest beta: {best_beta:.3f} with loss {best_loss:.4f}")

if __name__ == "__main__":
    beta_vals = np.arange(0.998, 0.898, -0.002)
    beta_vals = [round(float(b), 3) for b in beta_vals]

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_beta_sweep')
    fix_seed(43)
    run_regression_experiment(beta_vals=beta_vals)
