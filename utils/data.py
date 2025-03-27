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

# Dataset name to number of classes mapping
DATASET_TO_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'imagenet': 1000,
    'arabic_characters': 28,
    'arabic_digits': 10,
    'beans': 3,
    'cub200': 200,
    'dtd': 47,
    'food101': 101,
    'fgvc_aircraft': 100,
    'flowers102': 102,
    'fashion_mnist': 10,
    'med_mnist/pathmnist': 9,
    'med_mnist/octmnist': 4,
    'med_mnist/dermamnist': 7,
    'celeb_a': 40,
    'dsprites': 1,
    'imagenette': 10,
    'imagenet100': 100,
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

def stratified_train_val_split(full_dataset, val_size):
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
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    train_idx, val_idx = next(sss.split(X, y))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    return train_subset, val_subset


def stratified_subset(dataset, n_samples):
    """
    Return a new Subset of 'dataset' with 'n_samples' selected via stratified sampling,
    ensuring that every class is present (assuming n_samples >= number_of_classes).
    """
    if n_samples > len(dataset):
        raise ValueError(f"Requested train_size={n_samples} exceeds dataset size {len(dataset)}.")

    labels = get_labels_from_subset(dataset)
    unique_labels = np.unique(labels)
    if n_samples < len(unique_labels):
        raise ValueError(
            f"train_size={n_samples} < number_of_classes={len(unique_labels)}; "
            f"cannot guarantee at least one sample per class."
        )

    X = np.arange(len(dataset))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
    sub_idx, _ = next(sss.split(X, labels))
    return Subset(dataset, sub_idx)


def get_data_loaders(dataset,
                     train_batch_size=500,
                     test_batch_size=500,
                     train_size=None,
                     test_size=None,
                     val_size=None,
                     num_workers=6,
                     transform_train=None,
                     transform_test=None):
    """
    Get train, test, and (optionally) val DataLoaders with guaranteed class coverage.
    If val_size is specified, the training set is stratified so that no class is dropped.
    If train_size is specified, the final training subset is again stratified to ensure coverage.
    """
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
        if transform_to_use in ['cifar10', 'cifar100', 'arabic_characters', 'arabic_digits']:
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
            'mnist','fashion_mnist','med_mnist/pathmnist',
            'med_mnist/octmnist','med_mnist/dermamnist','dsprites']:
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
            'imagenet','fgvc_aircraft','places365_small','flowers102',
            'beans','cub200','dtd','food101','celeb_a','imagenette','imagenet100']:
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
    if dataset_to_use == 'cifar10':
        train_full = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    elif dataset_to_use == 'cifar100':
        train_full = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)

    elif dataset_to_use == 'mnist':
        train_full = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)

    elif dataset_to_use == 'imagenet':
        # Typically, we only have a "val" set for ImageNet if we don't have user-provided train
        train_full = None  # or load from somewhere else
        test_set = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform_test)

    elif dataset_to_use in ['arabic_characters','fashion_mnist','arabic_digits',
                            'cub200','food101','dsprites','imagenette']:
        hf_trainset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="train",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="test",
            trust_remote_code=True
        )
        train_full = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set   = HuggingFaceDataset(hf_testset, transform=transform_test)

    elif dataset_to_use in ['imagenet100']:
        hf_trainset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="train",
            trust_remote_code=True,
            cache_dir="/users/hleyang/scratch/cache"
        )
        hf_testset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="validation",
            trust_remote_code=True,
            cache_dir="/users/hleyang/scratch/cache"
        )
        train_full = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set   = HuggingFaceDataset(hf_testset, transform=transform_test)

    elif dataset_to_use in ['fgvc_aircraft','flowers102','beans','dtd','celeb_a']:
        hf_train_dataset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="train",
            trust_remote_code=True
        )
        hf_val_dataset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="validation",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            f"./utils/aidatasets/images/{dataset_to_use}.py",
            split="test",
            trust_remote_code=True
        )
        hf_trainset = datasets.concatenate_datasets([hf_train_dataset, hf_val_dataset])
        train_full  = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set    = HuggingFaceDataset(hf_testset, transform=transform_test)

    elif dataset_to_use in ['med_mnist/pathmnist','med_mnist/octmnist','med_mnist/dermamnist']:
        name = dataset_to_use.split('/')[-1]
        hf_train_dataset = datasets.load_dataset(
            "./utils/aidatasets/images/med_mnist.py",
            name=name,
            split="train",
            trust_remote_code=True
        )
        hf_val_dataset = datasets.load_dataset(
            "./utils/aidatasets/images/med_mnist.py",
            name=name,
            split="validation",
            trust_remote_code=True
        )
        hf_testset = datasets.load_dataset(
            "./utils/aidatasets/images/med_mnist.py",
            name=name,
            split="test",
            trust_remote_code=True
        )
        hf_trainset = datasets.concatenate_datasets([hf_train_dataset, hf_val_dataset])
        train_full  = HuggingFaceDataset(hf_trainset, transform=transform_train)
        test_set    = HuggingFaceDataset(hf_testset, transform=transform_test)

    else:
        raise NotImplementedError(f"The specified dataset '{dataset_to_use}' is not implemented.")

    # Split the dataset if val_size is specified
    if (train_full is not None) and (val_size is not None):
        if val_size == -1:
            val_size = test_size if test_size is not None else len(test_set)
        if val_size >= len(train_full):
            raise ValueError("Validation size should be smaller than the original training set size.")
        # We do a stratified split so that the new "train" portion
        # still has every label after splitting off val_size samples.
        train_full, val_subset = stratified_train_val_split(train_full, val_size)
        val_loader = DataLoader(
            val_subset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = None

    # Subsample the Train Set
    if (train_full is not None) and (train_size is not None):
        train_full = stratified_subset(train_full, train_size)

    # Subsample the Test Set (If test_size is specified)
    if (test_size is not None) and (test_size < len(test_set)):
        # For test, you *could* do a stratified_subset if you wish to preserve coverage,
        # but here we'll just do a random subset as the original code did.
        indices = np.random.choice(len(test_set), test_size, replace=False)
        test_set = Subset(test_set, indices)
    elif (test_size is not None) and (test_size >= len(test_set)):
        raise ValueError("test_size >= size of test set.")

    # Create DataLoaders
    if (dataset_to_use == 'imagenet') or (train_full is None):
        train_loader = None
    else:
        train_loader = DataLoader(
            train_full,
            batch_size=train_batch_size,
            shuffle=True,  # still shuffle in each epoch
            num_workers=num_workers
        )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, val_loader
