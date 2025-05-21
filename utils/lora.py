import torch
from torch import nn as nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """
    A Linear layer that applies LoRA to a frozen, pretrained Linear.
    """

    def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
        """
        :param original_layer: The pretrained, frozen nn.Linear layer
        :param r: Rank of the LoRA decomposition
        :param alpha: Scaling factor for LoRA
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = r
        self.alpha = alpha

        # Freeze the original layer's parameters
        self.weight = nn.Parameter(original_layer.weight.data, requires_grad=False)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data, requires_grad=False)
        else:
            self.bias = None

        # LoRA parameters B and A
        # B: [out_features, r]
        # A: [r, in_features]
        self.B = nn.Parameter(torch.zeros((self.out_features, r)))
        self.A = nn.Parameter(torch.zeros((r, self.in_features)))

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        nn.init.zeros_(self.A)

    def forward(self, x):
        # Normal forward with the frozen weight
        result = F.linear(x, self.weight, self.bias)

        # LoRA path: B @ A
        # shape of BA = [out_features, in_features]
        # Then F.linear with BA
        lora_update = F.linear(x, (self.alpha / self.r) * (self.B @ self.A))

        return result + lora_update


class LoRAConv2d(nn.Module):
    """
    A Conv2d layer that applies LoRA to a frozen, pretrained Conv2d.
    """

    def __init__(self, original_layer: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        """
        :param original_layer: The pretrained, frozen nn.Conv2d layer
        :param r: Rank of the LoRA decomposition
        :param alpha: Scaling factor for LoRA
        """
        super().__init__()

        self.out_channels = original_layer.out_channels
        self.in_channels = original_layer.in_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        self.bias_available = (original_layer.bias is not None)

        self.r = r
        self.alpha = alpha

        # Freeze original parameters
        self.weight = nn.Parameter(original_layer.weight.data, requires_grad=False)
        if self.bias_available:
            self.bias = nn.Parameter(original_layer.bias.data, requires_grad=False)
        else:
            self.bias = None

        # Flattened shape for weight is [out_channels, in_channels * k_h * k_w]
        k_h, k_w = self.kernel_size
        fan_in = self.in_channels * k_h * k_w  # Flattened input dim

        # Define LoRA parameters: B and A
        # B: [out_channels, r]
        # A: [r, fan_in]
        self.B = nn.Parameter(torch.zeros((self.out_channels, r)))
        self.A = nn.Parameter(torch.zeros((r, fan_in)))

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        nn.init.zeros_(self.A)

    def forward(self, x):
        # Standard (frozen) convolution
        original_out = F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        # Compute LoRA update
        # 1) Flatten conv kernel in the same manner as above
        # 2) Multiply B and A -> shape [out_channels, in_channels * k_h * k_w]
        # 3) Reshape it back to [out_channels, in_channels, k_h, k_w]

        BA = self.B @ self.A  # shape [out_channels, fan_in]

        # Reshape to conv kernel
        k_h, k_w = self.kernel_size
        lora_weight = BA.view(
            self.out_channels,
            self.in_channels,
            k_h,
            k_w
        ) * (self.alpha / self.r)  # scale by alpha

        # Perform conv2d with the LoRA weight (no extra bias term for LoRA)
        lora_out = F.conv2d(
            x,
            lora_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return original_out + lora_out


def get_lora_model(model: nn.Module, r: int = 4, alpha: float = 1.0):
    """
    Recursively replace all Conv2d and Linear modules in model with LoRA-enabled versions.
    Freezes original weights and adds LoRA parameters.
    """
    for name, child in list(model.named_children()):
        # If child is a Conv2d, replace it with LoRAConv2d
        if isinstance(child, nn.Conv2d):
            lora_module = LoRAConv2d(child, r=r, alpha=alpha)
            setattr(model, name, lora_module)

        # If child is a Linear, replace it with LoRALinear
        elif isinstance(child, nn.Linear):
            lora_module = LoRALinear(child, r=r, alpha=alpha)
            setattr(model, name, lora_module)

        else:
            # Recursively traverse children
            get_lora_model(child, r=r, alpha=alpha)

    return model
