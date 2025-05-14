"""
This file plots our activation function for different values of beta.
"""
import os
import torch
import matplotlib.pyplot as plt
from utils.curvature_tuning import TrainableCTU


# Create a set of x-values over which to evaluate the activation
x_vals = torch.linspace(-0.6, 0.6, steps=200)

# Plot for several beta values
plt.figure(figsize=(7, 5))
betas = [0.5, 0.9]

cmap = plt.colormaps["Dark2"]
colors = [cmap(0), cmap(1), cmap(2)]

for idx, b in enumerate(betas):
    # Instantiate CT with the current beta
    activation = TrainableCTU(beta=b)

    # Forward pass: compute the output for all x_vals
    with torch.no_grad():  # no need for gradients when just plotting
        y_vals = activation(x_vals)

    plt.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], label=r"$\beta$" + f"={b:.1f}", linewidth=5, alpha=0.6)

# Also plot ReLU for comparison
relu = torch.nn.ReLU()
with torch.no_grad():
    y_relu = relu(x_vals)
plt.plot(x_vals.numpy(), y_relu.numpy(), color=colors[len(betas)], label=r"$\beta$" + f"=1.0 (ReLU)", linewidth=5, alpha=0.6)

fontsize = 20
plt.legend(fontsize=fontsize)
plt.xticks([-0.6, -0.3, 0, 0.3, 0.6], fontsize=fontsize)
plt.yticks([0, 0.15, 0.3, 0.45, 0.6], fontsize=fontsize)
plt.grid(True)
os.makedirs('./figures', exist_ok=True)
plt.savefig("./figures/demo_act.svg", bbox_inches="tight")
plt.show()
