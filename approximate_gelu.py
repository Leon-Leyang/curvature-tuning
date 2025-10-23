import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.curvature_tuning import TCTU


# Initialize
ctu = TCTU(num_input_dims=2, out_channels=1, raw_beta=0.5322, raw_coeff=10)
gelu = nn.GELU()
optimizer = torch.optim.Adam(ctu.parameters(), lr=1e-3)

# Input domain
x = torch.linspace(-10, 10, 10000).reshape(-1, 1)

# Ground truth
target = gelu(x)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    pred = ctu(x)
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Beta: {ctu.beta.item():.4f}, Coeff: {ctu.coeff.item():.4f}")

# Final values
print(f"Final raw_beta: {ctu._raw_beta.item():.4f}, raw_coeff: {ctu._raw_coeff.item():.4f}")
print(f"Final beta: {ctu.beta.item():.4f}, coeff: {ctu.coeff.item():.4f}")

# Plotting
with torch.no_grad():
    plt.plot(x.numpy(), target.numpy(), label='GELU')
    plt.plot(x.numpy(), ctu(x).numpy(), label='TrainableCTU')
    plt.legend()
    plt.title("CTU Approximation of GELU")
    plt.grid(True)
    plt.show()
