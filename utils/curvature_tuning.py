"""
This file implements the CTU (Curvature Tuning Unit) for CT and Trainable CTU.
It also provides utility functions to replace ReLU with CTU in a model and to compute the mean of beta and coeff parameters.
"""
import torch
from torch import nn
import torch.nn.functional as F


class CTU(nn.Module):
    """
    CTU for CT.
    """
    def __init__(self, shared_raw_beta, shared_raw_coeff, threshold=20):
        super().__init__()
        self.threshold = threshold
        self._raw_beta = shared_raw_beta
        self._raw_coeff = shared_raw_coeff
        self._raw_beta.requires_grad = False
        self._raw_coeff.requires_grad = False

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = torch.sigmoid(self._raw_coeff)
        one_minus_beta = 1 - beta
        x_scaled = x / one_minus_beta

        return (coeff * torch.sigmoid(beta * x_scaled) * x +
                (1 - coeff) * F.softplus(x_scaled, threshold=self.threshold) * one_minus_beta)


class TrainableCTU(nn.Module):
    """
    CTU for Trainable CT.
    """
    def __init__(self, num_input_dims, out_channels, raw_beta=1.386, raw_coeff=0.0, threshold=20):
        super().__init__()
        self.threshold = threshold

        # Decide channel dim based on input shape
        if num_input_dims == 2 or num_input_dims == 3:  # (B, C) or (B, L, D)
            channel_dim = -1
        elif num_input_dims == 4: # (B, C, H, W)
            channel_dim = 1
        else:
            raise NotImplementedError(f"Unsupported input dimension {num_input_dims}")

        param_shape = [1] * num_input_dims
        param_shape[channel_dim] = out_channels

        # Init beta
        self._raw_beta = nn.Parameter(torch.full(param_shape, float(raw_beta)))

        # Init coeff
        self._raw_coeff = nn.Parameter(torch.full(param_shape, float(raw_coeff)))

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = torch.sigmoid(self._raw_coeff)
        one_minus_beta = 1 - beta
        x_scaled = x / one_minus_beta

        return (coeff * torch.sigmoid(beta * x_scaled) * x +
                (1 - coeff) * F.softplus(x_scaled, threshold=self.threshold) * one_minus_beta)


def replace_module(model, old_module=nn.ReLU, new_module=CTU, **kwargs):
    """
    Replace all instances of old_module in the model with new_module.
    """
    device = next(model.parameters(), torch.tensor([])).device  # Handle models with no parameters

    # Replace modules
    for name, module in model.named_modules():
        if isinstance(module, old_module):
            ct = new_module(**kwargs).to(device)

            # Replace module in the model
            names = name.split(".")
            parent = model
            for n in names[:-1]:
                if n.isdigit():
                    parent = parent[int(n)]  # for Sequential/ModuleList
                else:
                    parent = getattr(parent, n)

            last_name = names[-1]
            if last_name.isdigit():
                parent[int(last_name)] = ct  # for Sequential/ModuleList
            else:
                setattr(parent, last_name, ct)

    return model


def replace_module_dynamic(model, input_shape, old_module=nn.ReLU, new_module=TrainableCTU, **kwargs):
    """
    Replace all instances of old_module in the model with new_module that is dynamically created based on the number of output channels.
    """
    device = next(model.parameters(), torch.tensor([])).device
    dummy_input = torch.randn(*input_shape).to(device)

    module_metadata = {}  # name -> (num_input_dims, out_channels)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            num_input_dims = input[0].dim()
            if num_input_dims in (2, 3):    # (B, C) or (B, L, D)
                out_channels = output.shape[-1]
            elif num_input_dims == 4:       # (B, C, H, W)
                out_channels = output.shape[1]
            else:
                raise NotImplementedError(f"Unsupported output shape {output.shape} in {name}")
            module_metadata[name] = (num_input_dims, out_channels)

        return hook

    # Register hooks to all modules of the target type
    for name, module in model.named_modules():
        if isinstance(module, old_module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run dummy forward pass
    model(dummy_input)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Replace modules
    for name, module in model.named_modules():
        if isinstance(module, old_module) and name in module_metadata:
            num_input_dims, out_channels = module_metadata[name]
            ct = new_module(num_input_dims=num_input_dims, out_channels=out_channels, **kwargs).to(device)

            # Replace module in the model
            names = name.split(".")
            parent = model
            for n in names[:-1]:
                if n.isdigit():
                    parent = parent[int(n)]  # for Sequential/ModuleList
                else:
                    parent = getattr(parent, n)

            last_name = names[-1]
            if last_name.isdigit():
                parent[int(last_name)] = ct  # for Sequential/ModuleList
            else:
                setattr(parent, last_name, ct)

    return model


def get_mean_beta_and_coeff(model):
    """
    Iterate through the model to compute the mean of beta and coeff parameters of TrainableCTU modules.
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

    if beta_vals and coeff_vals:
        all_beta = torch.cat(beta_vals)
        all_coeff = torch.cat(coeff_vals)
        mean_beta = all_beta.mean().item()
        mean_coeff = all_coeff.mean().item()
        return mean_beta, mean_coeff
    else:
        return None, None
