"""
This file implements the Curvature Tuning (CT) activation function.
A utility function is also provided to replace all instances of a module in a model with a new module (e.g., ReLU with CT).
"""
import torch
from torch import nn
import torch.nn.functional as F


class CT(nn.Module):
    """
    Curvature Tuning (CT) activation function.
    This activation function is designed to be used in place of ReLU.
    The computation is defined as:
    CT(x) = coeff * sigmoid(beta * x / (1 - beta)) * x +
             (1 - coeff) * softplus(x / (1 - beta)) * (1 - beta)
    """
    def __init__(self, num_input_dims, out_channels, raw_beta=10.0, raw_coeff=0.0, threshold=20):
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
        self._raw_beta = nn.Parameter(torch.full(param_shape, raw_beta))

        # Init coeff
        self._raw_coeff = nn.Parameter(torch.full(param_shape, raw_coeff))

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        return (self.coeff * torch.sigmoid(self.beta * x / (1 - self.beta)) * x +
                (1 - self.coeff) * F.softplus(x / (1 - self.beta), threshold=self.threshold) * (1 - self.beta))


class ReplacementMapping:
    def __init__(self, old_module, new_module, **kwargs):
        self.old_module = old_module
        self.new_module = new_module
        self.kwargs = kwargs

    def __call__(self, module):
        if isinstance(module, self.old_module):
            return self.new_module(**self.kwargs)
        return module


def replace_module(model, old_module=nn.ReLU, new_module=CT, **kwargs):
    if not isinstance(model, nn.Module):
        raise ValueError("Expected model to be an instance of torch.nn.Module")

    replacement_mapping = ReplacementMapping(old_module, new_module, **kwargs)

    device = next(model.parameters(), torch.tensor([])).device  # Handle models with no parameters

    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(module).to(device)

        # Traverse module hierarchy to assign new module
        module_names = name.split(".")
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)

    return model


def replace_module_per_channel(model, input_shape, old_module=nn.ReLU, new_module=CT, **kwargs):
    """
    Safely replace all modules in a model with per-channel new modules.

    For each old module, we record the number of input dimensions and the number of output channels
    via a forward hook before replacing.

    Args:
        model (nn.Module): the model to modify in-place.
        input_shape (tuple): dummy input shape for tracing.
        old_module (type): module type to replace (default: nn.ReLU).
        new_module (type): module class to instantiate (must accept num_input_dims and out_channels).
        **kwargs: additional arguments for the new module constructor.
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
    Iterate through the model to compute the mean of beta and coeff parameters.
    """
    beta_vals = []
    coeff_vals = []

    for module in model.modules():
        if isinstance(module, CT):
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
