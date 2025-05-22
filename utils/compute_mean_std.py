"""
This file is for computing the mean and standard deviation of image datasets.
"""
import datasets
import torch
from torchvision import transforms
from tqdm import tqdm


def compute_dataset_mean_std(dataset, image_key="image"):
    """
    Compute the mean and standard deviation of a dataset with a progress bar.

    Args:
        dataset: Hugging Face `datasets.Dataset` object.
        image_key (str): Key in the dataset dictionary that contains the image data.

    Returns:
        tuple: Mean and standard deviation as lists for each channel (4 decimal places).
    """
    # Define a transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to [C, H, W] with values in range [0, 1]
    ])

    # Initialize variables to accumulate pixel values
    sum_pixels = torch.zeros(3)  # For RGB channels
    sum_squared_pixels = torch.zeros(3)
    num_pixels = 0

    # Iterate through the dataset with a progress bar
    for example in tqdm(dataset, desc="Computing Mean and Std"):
        # Convert the image to a tensor
        image_tensor = transform(example[image_key])

        # Update accumulators
        sum_pixels += image_tensor.sum(dim=(1, 2))  # Sum across height and width
        sum_squared_pixels += (image_tensor ** 2).sum(dim=(1, 2))
        num_pixels += image_tensor.size(1) * image_tensor.size(2)  # Height * Width

    # Compute mean and std
    mean = sum_pixels / num_pixels
    std = torch.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))

    # Round to 4 decimal places
    mean = [round(x.item(), 3) for x in mean]
    std = [round(x.item(), 3) for x in std]

    return mean, std


def main(ds_name):
    # Replace with the path to your dataset script and the split name
    dataset_split = "train"
    image_key = "image"

    # Load the dataset
    dataset = datasets.load_dataset(ds_name, split=dataset_split, trust_remote_code=True)
    # dataset = datasets.load_dataset(dataset_name, split=dataset_split, name=variant, trust_remote_code=True)

    # Compute mean and standard deviation
    mean, std = compute_dataset_mean_std(dataset, image_key=image_key)
    print(f"Dataset: {ds_name}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


# Example usage
if __name__ == "__main__":
    ds = ["randall-lab/beans"]
    for dataset in ds:
        main(dataset)
