"""
This file contains utility functions for loading datasets.
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import transforms as transforms
import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils import fix_seed


# Predefined normalization values for different datasets
NORMALIZATION_VALUES = {
    'cifar10': ([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
    'cifar100': ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
    'mnist': ([0.131, 0.131, 0.131], [0.308, 0.308, 0.308]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'arabic-characters': ([0.101, 0.101, 0.101], [0.301, 0.301, 0.301]),
    'arabic-digits': ([0.165, 0.165, 0.165], [0.371, 0.371, 0.371]),
    'beans': ([0.485, 0.518, 0.313], [0.211, 0.223, 0.2]),
    'cub200': ([0.486, 0.5, 0.433], [0.232, 0.228, 0.267]),
    'dtd': ([0.531, 0.475, 0.427], [0.271, 0.263, 0.271]),
    'food101': ([0.549, 0.445, 0.344], [0.273, 0.276, 0.28]),
    'fgvc-aircraft': ([0.485, 0.52, 0.548], [0.219, 0.21, 0.241]),
    'flowers102': ([0.43, 0.38, 0.295], [0.295, 0.246, 0.273]),
    'fashion-mnist': ([0.286, 0.286, 0.286], [0.353, 0.353, 0.353]),
    'medmnist/pathmnist': ([0.741, 0.533, 0.706], [0.124, 0.177, 0.124]),
    'medmnist/octmnist': ([0.189, 0.189, 0.189], [0.196, 0.196, 0.196]),
    'medmnist/dermamnist': ([0.763, 0.538, 0.561], [0.137, 0.154, 0.169]),
    'celeb-a': ([0.506, 0.426, 0.383], [0.311, 0.29, 0.29]),
    'dsprites': ([0.0, 0.0, 0.0], [0.001, 0.001, 0.001]),
    'imagenette': ([0.459, 0.455, 0.429], [0.286, 0.282, 0.305]),
}

# Dataset name to number of classes mapping
DATASET_TO_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'imagenet': 1000,
    'arabic-characters': 28,
    'arabic-digits': 10,
    'beans': 3,
    'cub200': 200,
    'dtd': 47,
    'food101': 101,
    'fgvc-aircraft': 100,
    'flowers102': 102,
    'fashion-mnist': 10,
    'medmnist/pathmnist': 9,
    'medmnist/octmnist': 4,
    'medmnist/dermamnist': 7,
    'celeb-a': 40,
    'dsprites': 1,
    'imagenette': 10,
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
        sample = self.dataset[int(idx)]
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


def get_labels_from_subset(ds):
    """
    Extracts all labels from a PyTorch Dataset or Subset (ds).
    Expects each sample to be in the form (image, label).
    """
    if isinstance(ds, Subset):
        # ds.indices are the indices into ds.dataset
        labels = [ds.dataset[int(idx)][1] for idx in ds.indices]
    else:
        # Plain dataset
        labels = [ds[i][1] for i in range(len(ds))]
    return np.array(labels)

def stratified_two_split(full_dataset, val_size, seed=42):
    """
    Perform a single stratified split on 'full_dataset' of size len(full_dataset).
    'val_size' should be the absolute number of samples for the val subset.
    Returns: (train_subset, val_subset)
    """
    n_samples = len(full_dataset)
    if val_size >= n_samples:
        raise ValueError(f"val_size={val_size} is >= dataset size={n_samples}.")

    X = np.arange(n_samples)
    y = get_labels_from_subset(full_dataset)

    # We want 'val_size' samples for val.
    # In StratifiedShuffleSplit, we typically specify fractions or absolute counts
    # by setting 'test_size'. So we do:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(X, y))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    return train_subset, val_subset


def stratified_subset(full_dataset, n_samples, seed=42):
    """
    Return a new Subset of 'dataset' with 'n_samples' selected via stratified sampling,
    ensuring that every class is present (assuming n_samples >= number_of_classes).
    """
    if n_samples > len(full_dataset):
        raise ValueError(f"Requested train_size={n_samples} exceeds dataset size {len(full_dataset)}.")

    labels = get_labels_from_subset(full_dataset)
    unique_labels = np.unique(labels)
    if n_samples < len(unique_labels):
        raise ValueError(
            f"train_size={n_samples} < number_of_classes={len(unique_labels)}; "
            f"cannot guarantee at least one sample per class."
        )

    X = np.arange(len(full_dataset))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    sub_idx, _ = next(sss.split(X, labels))
    return Subset(full_dataset, sub_idx)


def get_data_loaders(dataset,
                     train_batch_size=128,
                     test_batch_size=1024,
                     train_size=None,
                     test_size=None,
                     val_size=None,
                     num_workers=6,
                     transform_train=None,
                     transform_test=None,
                     seed=42):
    """
    Get train, test, and val DataLoaders.
    """
    fix_seed(seed)

    # Identify which transformations to apply based on dataset name
    if '_to_' in dataset:  # e.g., cifar10_to_cifar100
        transform_to_use = dataset.split('_to_')[0]
        dataset_to_use = dataset.split('_to_')[-1]
        normalization_to_use = dataset.split('_to_')[-1]
    else:
        transform_to_use = dataset
        dataset_to_use = dataset
        normalization_to_use = dataset

    # If transform_train/test not specified, define some defaults:
    if transform_train is None and transform_test is None:
        if transform_to_use in ['cifar10', 'cifar100', 'arabic-characters', 'arabic-digits']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
        elif transform_to_use in [
            'mnist','fashion-mnist','medmnist/pathmnist',
            'medmnist/octmnist','medmnist/dermamnist','dsprites']:
            transform_train = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])
        elif transform_to_use in [
            'imagenet','fgvc-aircraft','places365-small','flowers102',
            'beans','cub200','dtd','food101','celeb-a','imagenette']:
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
                transforms.Lambda(replicate_if_needed),
                transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
            ])

    # Load Dataset
    train_set, val_set, test_set = None, None, None
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
        # Only use the validation set for testing
        test_set = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform_test)

    elif dataset_to_use in ['arabic-characters','fashion-mnist','arabic-digits',
                            'cub200','food101','dsprites','imagenette']:
        hf_trainset = datasets.load_dataset(
            f"randall-lab/{dataset_to_use}",
            split="train",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            f"randall-lab/{dataset_to_use}",
            split="test",
            trust_remote_code=True
        )
        train_set = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set  = HuggingFaceDataset(hf_testset, transform=transform_test)

    elif dataset_to_use in ['fgvc-aircraft','flowers102','beans','dtd','celeb-a']:
        hf_trainset = datasets.load_dataset(
            f"randall-lab/{dataset_to_use}",
            split="train",
            trust_remote_code=True
        )
        hf_valset = datasets.load_dataset(
            f"randall-lab/{dataset_to_use}",
            split="validation",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            f"randall-lab/{dataset_to_use}",
            split="test",
            trust_remote_code=True
        )
        train_set  = HuggingFaceDataset(hf_trainset, transform=transform_train)
        val_set    = HuggingFaceDataset(hf_valset, transform=transform_test)
        test_set    = HuggingFaceDataset(hf_testset, transform=transform_test)

    elif dataset_to_use in ['medmnist/pathmnist','medmnist/octmnist','medmnist/dermamnist']:
        name = dataset_to_use.split('/')[-1]
        hf_trainset = datasets.load_dataset(
            "randall-lab/medmnist",
            name=name,
            split="train",
            trust_remote_code=True
        )
        hf_valset = datasets.load_dataset(
            "randall-lab/medmnist",
            name=name,
            split="validation",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            "randall-lab/medmnist",
            name=name,
            split="test",
            trust_remote_code=True
        )
        train_set  = HuggingFaceDataset(hf_trainset, transform=transform_train)
        val_set    = HuggingFaceDataset(hf_valset, transform=transform_test)
        test_set   = HuggingFaceDataset(hf_testset, transform=transform_test)

    else:
        raise NotImplementedError(f"The specified dataset '{dataset_to_use}' is not implemented.")

    if train_set is not None and val_set is None:
        # If val_set is not specified, split the train_set into train and val
        default_val_size = int(0.2 * len(train_set))
        train_set, val_set = stratified_two_split(train_set, default_val_size, seed)

    # Subsample the Train Set (If train_size is specified)
    if train_set is not None:
        if train_size is not None:
            if train_size <= len(train_set):
                train_set = stratified_subset(train_set, train_size, seed)
            else:
                raise ValueError("train_size > size of train set.")

        # Subsample the Val Set (If val_size is specified)
        if val_size is not None:
            if val_size <= len(val_set):
                val_set = stratified_subset(val_set, val_size, seed)
            else:
                raise ValueError("val_size > size of val set.")

    # Subsample the Test Set (If test_size is specified)
    if test_size is not None:
        if test_size <= len(test_set):
            test_set = stratified_subset(test_set, test_size, seed)
        else:
            raise ValueError("test_size > size of test set.")

    # Create DataLoaders
    if train_set is None:
        train_loader = None
        val_loader = None
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,  # still shuffle in each epoch
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, val_loader
