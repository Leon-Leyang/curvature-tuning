import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import imageio
from utils.utils import MLP, get_file_name, fix_seed, set_logger, get_log_file_path
from utils.curvature_tuning import replace_module_dynamic, TrainableCTU
from utils.lora import get_lora_model
from loguru import logger


def generate_circular_data(n_points):
    n_points_small_circle = int(n_points / 4)
    n_points_large_circle = n_points - n_points_small_circle

    radius_inner = 1.0
    angles_inner = np.linspace(0, 2 * np.pi, n_points_small_circle, endpoint=False)
    x_inner = radius_inner * np.cos(angles_inner)
    y_inner = radius_inner * np.sin(angles_inner)
    labels_inner = np.zeros(n_points_small_circle, dtype=int)

    radius_outer = 3
    angles_outer = np.linspace(0, 2 * np.pi, n_points_large_circle, endpoint=False)
    x_outer = radius_outer * np.cos(angles_outer)
    y_outer = radius_outer * np.sin(angles_outer)
    labels_outer = np.ones(n_points_large_circle, dtype=int)

    X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
    y = np.concatenate((labels_inner, labels_outer))
    return X, y


def render_frame(points, target, xx, yy, pred, mesh_dim, color):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors_map = ["olive" if t == 0 else "palevioletred" for t in target.cpu().numpy()]
    ax.scatter(points[:, 0].cpu(), points[:, 1].cpu(), c=colors_map, alpha=0.6, edgecolors="none")
    ax.contour(xx, yy, pred[:, 0].reshape((mesh_dim, mesh_dim)), levels=[0], colors=[color], linewidths=[5])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def train_with_frames(model, points, target, training_steps, grid, xx, yy, mesh_dim, color):
    frames = []
    optim = torch.optim.AdamW(model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)

    for step in range(training_steps):
        output = model(points)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if step % 100 == 0 or step == training_steps - 1:
            with torch.no_grad():
                pred = model(grid).cpu().numpy()
            frame = render_frame(points, target, xx, yy, pred, mesh_dim, color)
            frames.append(frame)
    return frames


def plot_classification_gif(width=10, depth=1, training_steps=4000, finetune_step=4000, init_beta=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    X, y = generate_circular_data(n_points=180)
    points = torch.from_numpy(X).float().to(device)
    target = torch.from_numpy(y).long().to(device)

    baseline_model = MLP(2, 1, depth, width, nn.ReLU()).to(device)
    optim = torch.optim.AdamW(baseline_model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.1)
    for step in range(training_steps):
        output = baseline_model(points)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()  # pretrain

    domain_bound = np.max(np.abs(X)) * 1.2
    mesh_dim = 400
    xx, yy = np.meshgrid(
        np.linspace(-domain_bound, domain_bound, mesh_dim),
        np.linspace(-domain_bound, domain_bound, mesh_dim),
    )
    grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().to(device)

    with torch.no_grad():
        pred_base = baseline_model(grid).cpu().numpy()
    cmap = plt.colormaps["Dark2"]
    colors = [cmap(2), cmap(1), cmap(0)]
    baseline_frame = render_frame(points, target, xx, yy, pred_base, mesh_dim, colors[0])

    os.makedirs('./figures', exist_ok=True)
    log_name = get_file_name(get_log_file_path())
    baseline_frame.save(f'./figures/{log_name}_base.png')

    lora_model = get_lora_model(copy.deepcopy(baseline_model), r=1, alpha=1).to(device)
    lora_frames = [baseline_frame] + train_with_frames(lora_model, points, target, finetune_step, grid, xx, yy, mesh_dim, colors[1])
    lora_frames[0].save(f'./figures/{log_name}_lora.gif', save_all=True, append_images=lora_frames[1:], duration=100, loop=0)

    ct_model = replace_module_dynamic(copy.deepcopy(baseline_model), (1, 2), old_module=nn.ReLU,
                                      new_module=TrainableCTU, raw_beta=torch.logit(torch.tensor(init_beta)).item()).to(device)
    ct_frames = [baseline_frame] + train_with_frames(ct_model, points, target, finetune_step, grid, xx, yy, mesh_dim, colors[2])
    ct_frames[0].save(f'./figures/{log_name}_ct.gif', save_all=True, append_images=ct_frames[1:], duration=100, loop=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--training_steps", type=int, default=4000)
    parser.add_argument("--finetune_step", type=int, default=4000)
    parser.add_argument("--init_beta", type=float, default=0.4)
    args = parser.parse_args()

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{args.width}_depth{args.depth}_beta{args.init_beta:.2f}')
    fix_seed(42)

    plot_classification_gif(width=args.width, depth=args.depth,
                            training_steps=args.training_steps,
                            finetune_step=args.finetune_step,
                            init_beta=args.init_beta)
