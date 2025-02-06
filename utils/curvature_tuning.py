import torch
from torch import nn


class BetaAgg(nn.Module):
    def __init__(self, beta=0, coeff=0.5, threshold=20, trainable=False):
        assert 0 <= beta < 1
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta.requires_grad_(trainable)
        self.coeff = coeff
        self.threshold = threshold

    def forward(self, x):
        beta = self.beta
        normal_ver = (
            self.coeff * torch.sigmoid(beta * x / (1 - beta)) * x +
            (1 - self.coeff) * torch.log(1 + torch.exp(x / (1 - beta))) * (1 - beta)
        )
        overflow_ver = (
            self.coeff * torch.sigmoid(beta * x / (1 - beta)) * x +
            (1 - self.coeff) * x
        )
        return torch.where(x / (1 - beta) <= self.threshold, normal_ver, overflow_ver)


class ReplacementMapping:
    def __init__(self, old_module, new_module, **kwargs):
        self.old_module = old_module
        self.new_module = new_module
        self.kwargs = kwargs

    def __call__(self, module):
        if isinstance(module, self.old_module):
            return self.new_module(**self.kwargs)
        return module


def replace_module(model, old_module=nn.ReLU, new_module=BetaAgg, **kwargs):
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
