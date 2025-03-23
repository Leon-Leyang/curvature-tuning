"""
This file contains utility functions for loading datasets.
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import transforms as transforms
import datasets


# Predefined normalization values for different datasets
NORMALIZATION_VALUES = {
    'cifar10': ([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
    'cifar100': ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
    'mnist': ([0.131, 0.131, 0.131], [0.308, 0.308, 0.308]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'arabic_characters': ([0.101, 0.101, 0.101], [0.301, 0.301, 0.301]),
    'arabic_digits': ([0.165, 0.165, 0.165], [0.371, 0.371, 0.371]),
    'beans': ([0.485, 0.518, 0.313], [0.211, 0.223, 0.2]),
    'cub200': ([0.486, 0.5, 0.433], [0.232, 0.228, 0.267]),
    'dtd': ([0.531, 0.475, 0.427], [0.271, 0.263, 0.271]),
    'food101': ([0.549, 0.445, 0.344], [0.273, 0.276, 0.28]),
    'fgvc_aircraft': ([0.485, 0.52, 0.548], [0.219, 0.21, 0.241]),
    'flowers102': ([0.43, 0.38, 0.295], [0.295, 0.246, 0.273]),
    'fashion_mnist': ([0.286, 0.286, 0.286], [0.353, 0.353, 0.353]),
    'med_mnist/pathmnist': ([0.741, 0.533, 0.706], [0.124, 0.177, 0.124]),
    'med_mnist/octmnist': ([0.189, 0.189, 0.189], [0.196, 0.196, 0.196]),
    'med_mnist/dermamnist': ([0.763, 0.538, 0.561], [0.137, 0.154, 0.169]),
    'celeb_a': ([0.506, 0.426, 0.383], [0.311, 0.29, 0.29]),
    'dsprites': ([0.0, 0.0, 0.0], [0.001, 0.001, 0.001]),
    'imagenette': ([0.459, 0.455, 0.429], [0.286, 0.282, 0.305]),
    'imagenet100': ([0.481, 0.453, 0.399], [0.277, 0.271, 0.281]),
}


class HuggingFaceDataset(torch.utils.data.Dataset):
    """
    Wrapper class for Hugging Face datasets.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def replicate_if_needed(x):
    """
    Replicate a single-channel tensor to three channels if needed.
    If the input has more than one channel, it is returned as is.
    """
    if x.shape[0] == 1:  # Check if there's only one channel
        return x.repeat(3, 1, 1)  # Replicate to 3 channels
    return x  # Return unchanged if already has more than 1 channel


def get_data_loaders(dataset, train_batch_size=500, test_batch_size=500, train_size=None, test_size=None, val_size=None, num_workers=6, transform_train=None, transform_test=None):
    """
    Get the data loaders for the dataset.
    """
    if '_to_' in dataset:  # e.g., cifar10_to_cifar100
        transform_to_use = dataset.split('_to_')[0]
        dataset_to_use = dataset.split('_to_')[-1]
        normalization_to_use = dataset.split('_to_')[-1]
    else:
        transform_to_use = dataset
        dataset_to_use = dataset
        normalization_to_use = dataset

    if transform_train is None and transform_test is None:
        if transform_to_use in ['cifar10', 'cifar100', 'arabic_characters', 'arabic_digits']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),  # Apply conditional replication
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),  # Apply conditional replication
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
        elif transform_to_use in ['mnist', 'fashion_mnist', 'med_mnist/pathmnist', 'med_mnist/octmnist',
                                  'med_mnist/dermamnist', 'dsprites']:
            transform_train = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),  # Apply conditional replication
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),  # Apply conditional replication
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
        elif transform_to_use in ['imagenet', 'fgvc_aircraft', 'places365_small', 'flowers102', 'beans', 'cub200',
                                  'dtd', 'food101', 'celeb_a', 'imagenette', 'imagenet100']:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),  # Apply conditional replication
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])

    if dataset_to_use == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'imagenet':
        train_set = None
        test_set = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform_test)
    elif dataset_to_use in ['arabic_characters', 'fashion_mnist', 'arabic_digits', 'cub200', 'food101', 'dsprites', 'imagenette']:
        hf_trainset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="train", trust_remote_code=True)
        hf_testset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="test", trust_remote_code=True)
        train_set = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set = HuggingFaceDataset(hf_testset, transform=transform_test)
    elif dataset_to_use in ['imagenet100']:
        hf_trainset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="train", trust_remote_code=True, cache_dir="/users/hleyang/scratch/cache")
        hf_testset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="validation", trust_remote_code=True, cache_dir="/users/hleyang/scratch/cache")
        train_set = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set = HuggingFaceDataset(hf_testset, transform=transform_test)
    elif dataset_to_use in ['fgvc_aircraft', 'flowers102', 'beans', 'dtd', 'celeb_a']:
        hf_train_dataset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="train", trust_remote_code=True)
        hf_val_dataset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="validation", trust_remote_code=True)
        hf_testset = datasets.load_dataset(f"./utils/aidatasets/images/{dataset_to_use}.py", split="test", trust_remote_code=True)
        hf_trainset = datasets.concatenate_datasets([hf_train_dataset, hf_val_dataset])
        train_set = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set = HuggingFaceDataset(hf_testset, transform=transform_test)
    elif dataset_to_use in ['med_mnist/pathmnist', 'med_mnist/octmnist', 'med_mnist/dermamnist']:
        name = dataset_to_use.split('/')[-1]
        hf_train_dataset = datasets.load_dataset("./utils/aidatasets/images/med_mnist.py", name=name, split="train", trust_remote_code=True)
        hf_val_dataset = datasets.load_dataset("./utils/aidatasets/images/med_mnist.py", name=name, split="validation", trust_remote_code=True)
        hf_testset = datasets.load_dataset("./utils/aidatasets/images/med_mnist.py", name=name, split="test", trust_remote_code=True)
        hf_trainset = datasets.concatenate_datasets([hf_train_dataset, hf_val_dataset])
        train_set = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set = HuggingFaceDataset(hf_testset, transform=transform_test)
    else:
        raise NotImplementedError(f'The specified dataset {dataset_to_use} is not implemented.')

    # Split the dataset if val_size is specified
    if val_size is not None:
        if val_size == -1:
            val_size = test_size if test_size is not None else len(test_set)
        if val_size >= len(train_set):
            raise ValueError("Validation size should be smaller than the original training set size.")

        remaining_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [remaining_size, val_size])

        val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_loader = None

    if train_size is not None:
        if train_size > len(train_set):
            raise ValueError("Requested train_size is larger than the remaining training set.")
        indices = np.random.choice(len(train_set), train_size, replace=False).tolist()
        train_set = Subset(train_set, indices)

    if test_size is not None:
        indices = np.random.choice(len(test_set), test_size, replace=False).tolist()
        test_set = Subset(test_set, indices)

    if dataset_to_use != 'imagenet':
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = None
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, val_loader
