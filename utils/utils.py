import os
import sys
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch import nn as nn
from utils.resnet import *
import numpy as np
from loguru import logger
import random


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


def get_pretrained_model(pretrained_ds='cifar100', device='cuda', model_name='resnet18'):
    """
    Get the pre-trained model.
    """
    name_to_model = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    name_to_model_imagenet = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
        'resnet152': torchvision.models.resnet152
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


def result_exists(ds, replace_then_lp=False, robustness_test=None):
    if replace_then_lp:
        ds = f'{ds}_replace_lp'
    log_file = get_log_file_path()
    if not os.path.exists(log_file):
        return False
    to_check = f'Best accuracy for {ds}:' if not robustness_test else f'Best robust accuracy for {ds} with {robustness_test} attack:'
    with open(log_file, 'r') as f:
        for line in f:
            if to_check in line:
                return True
    return False


def set_logger(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """
    Get the logger.
    The logger will be appended to a log file if it already exists.
    """
    os.makedirs("./logs", exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=print_level)
    log_file_path = f"./logs/{name}.log"
    logger.add(
        log_file_path,
        level=logfile_level,
        mode="a"  # Append mode
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


def plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, robust_config=None):
    if '/' in dataset:      # Hack to handle med_mnist/pathmnist
        dataset = dataset.split('/')[0]

    # Plot the test accuracy vs beta values
    plt.figure(figsize=(12, 8))
    plt.plot(beta_list, acc_list)
    plt.axhline(y=base_acc, color='r', linestyle='--', label='ReLU Test Accuracy')
    plt.xlabel('Beta')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Beta Values')

    # Ensure that both x-axis and y-axis show raw numbers without offset or scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    plt.xticks(beta_list[::5], rotation=45)
    plt.legend()
    os.makedirs('./figures', exist_ok=True)
    if robust_config:
        output_path = f"./figures/{get_file_name(get_log_file_path())}_{dataset}_{model_name}_{robust_config}.png"
    else:
        output_path = f"./figures/{get_file_name(get_log_file_path())}_{dataset}_{model_name}.png"
    plt.savefig(output_path)
