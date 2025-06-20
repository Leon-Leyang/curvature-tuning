"""
This file is for utility functions used across the project.
"""
import os
import sys
import torchvision
from torchvision.models import swin_t, swin_s
from utils.resnet_from_pytorch import resnet50_from_pytorch, resnet101_from_pytorch, resnet152_from_pytorch
from utils.model import *
import numpy as np
from loguru import logger
import random
from utils.curvature_tuning import replace_module
import json


class MLP(nn.Module):
    """
    A simple MLP for binary classification.
    """
    def __init__(self, in_features: int, out_features: int, depth: int, width: int, nonlinearity: nn.Module):
        super().__init__()
        self.register_buffer("depth", torch.as_tensor(depth))
        self.layer0 = torch.nn.Linear(in_features, width)
        for i in range(1, depth):
            setattr(
                self,
                f"layer{i}",
                nn.Linear(width, width),
            )
        self.output_layer = nn.Linear(width, out_features)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        for i in range(self.depth):
            x = getattr(self, f"layer{i}")(x)
            x = self.nonlinearity(x)
        x = self.output_layer(x)
        return x


def get_pretrained_model(pretrained_ds='cifar100', model_name='resnet18'):
    """
    Get the pre-trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    name_to_model = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'swin_t': swin_t,
        'swin_s': swin_s
    }
    name_to_model_imagenet = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': resnet50_from_pytorch,
        'resnet101': resnet101_from_pytorch,
        'resnet152': resnet152_from_pytorch
    }

    ckpt_folder = './ckpts'
    if 'cifar' in pretrained_ds:
        num_classes = 100 if 'cifar100' in pretrained_ds else 10
        model = name_to_model[model_name](num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'{model_name}_{pretrained_ds}_epoch200.pth'), weights_only=True))
    elif 'mnist' in pretrained_ds:
        num_classes = 10
        model = name_to_model[model_name](num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'{model_name}_{pretrained_ds}_epoch10.pth'), weights_only=True))
    elif pretrained_ds == 'imagenette':
        num_classes = 10
        if model_name == 'swin_t' or model_name == 'swin_s':
            model = name_to_model[model_name](num_classes=num_classes).to(device)
            model.head = nn.Linear(model.head.in_features, num_classes)
            model = replace_module(model, old_module=nn.GELU, new_module=nn.ReLU)
            model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'{model_name}_{pretrained_ds}_epoch200.pth'), weights_only=True))
        else:
            model = name_to_model[model_name](num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'{model_name}_{pretrained_ds}_epoch200.pth'), weights_only=True))
    elif pretrained_ds == 'imagenet':
        model = name_to_model_imagenet[model_name](weights='IMAGENET1K_V1').to(device)

    return model


def fix_seed(seed=42):
    """
    Fix the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_file_name(calling_file):
    """
    Returns the file name of the calling file without the extension.
    """
    file_name = os.path.basename(calling_file)
    return os.path.splitext(file_name)[0]


def set_logger(log_dir='./logs', print_level="INFO", logfile_level="DEBUG", name: str = None):
    """
    Get the logger.
    The logger will be appended to a log file if it already exists.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=print_level)
    log_file_path = f"{log_dir}/{name}.log"
    logger.add(
        log_file_path,
        level=logfile_level,
        mode="w"  # Overwrite mode
    )
    return log_file_path


def get_log_file_path():
    """
    Retrieve the path of the file the logger is writing to.
    """
    file_paths = []
    for handler in logger._core.handlers.values():
        sink = handler._sink
        # Check if the sink is a file and get its path
        if hasattr(sink, "_path"):
            file_paths.append(sink._path)
    assert len(file_paths) == 1, "Only one file-based log handler is supported."
    return file_paths[0]


def save_result_json(file_path, num_params, acc, **kwargs):
    data = {
        'num_params': num_params,
        'accuracy': acc,
    }
    data.update(kwargs)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
