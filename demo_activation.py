import torch
import matplotlib.pyplot as plt
from matplotlib.cm import magma
from matplotlib.colors import BoundaryNorm, ListedColormap
from utils.curvature_tuning import CTU
import numpy as np

# Setup
betas = torch.arange(0, 1.01, 0.1)
x = torch.linspace(-3, 3, 500)
coeffs = [1.0, 0.0, 0.5]
titles = ["Smoothing the Region Assignment", "Smoothing the Max", "Combined"]

# Reversed truncated colormap: color = 0.9 - 0.9 * beta
base_cmap = magma
truncated_colors = [base_cmap(0.9 - 0.9 * beta.item()) for beta in betas]
truncated_cmap = ListedColormap(truncated_colors)
boundaries = np.linspace(-0.05, 1.05, len(betas) + 1)
norm = BoundaryNorm(boundaries, truncated_cmap.N)

# Create figure and subplots with square plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # Adjust figure size to enforce square subplots

for ax, coeff, title in zip(axes, coeffs, titles):
    for beta in betas:
        raw_beta = torch.tensor(float('-inf') if beta == 0 else (float('inf') if beta == 1 else torch.logit(torch.tensor(beta))))
        raw_coeff = torch.tensor(float('-inf') if coeff == 0 else (float('inf') if coeff == 1 else torch.logit(torch.tensor(coeff))))
        activation = CTU(shared_raw_beta=raw_beta, shared_raw_coeff=raw_coeff)
        y = activation(x)
        color = base_cmap(0.9 - 0.9 * beta.item())
        ax.plot(x, y, color=color, linewidth=1)

    ax.set_title(title, fontsize=18)
    ax.set_aspect('equal')  # Enforce 1:1 aspect ratio strictly
    ax.set_box_aspect(1)   # Set box aspect to force square subplot
    ax.set_xlim(-3, 3)      # Explicitly set x range from -3 to 3
    ax.set_xticks(np.arange(-3, 4, 1))  # Set x ticks with interval 1
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=18)  # Set x and y tick label size

# Adjust layout to fit colorbar and equalize side margins
plt.subplots_adjust(left=0.04, right=0.9)

# Add vertical segmented colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = plt.colorbar(
    plt.cm.ScalarMappable(cmap=truncated_cmap, norm=norm),
    cax=cbar_ax,
    ticks=betas,
    boundaries=boundaries,
    spacing='uniform'
)
cb.set_label(r'$\beta$ values', fontsize=18)
cb.ax.tick_params(labelsize=18)

plt.savefig("./figures/activation_functions.pdf")
plt.show()