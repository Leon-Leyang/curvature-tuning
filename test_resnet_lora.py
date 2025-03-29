"""
This file is for counting the number of trainable parameters in a ResNet with or without LoRA.
"""
from torchvision.models import *
from utils.lora import get_lora_cnn
from utils.utils import count_trainable_parameters


if __name__ == "__main__":
    model = resnet50()

    # LoRA rank and alpha
    lora_rank = 4
    lora_alpha = 1.0

    # Count how many parameters are trainable
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters (Original): {trainable_params:,}")

    # Replace all Conv2d/Linear modules with LoRA-wrapped versions
    get_lora_cnn(model, r=lora_rank, alpha=lora_alpha)

    # Count how many parameters are trainable
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"LoRA rank = {lora_rank}")
    print(f"Total parameters in model: {total_params:,}")
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
