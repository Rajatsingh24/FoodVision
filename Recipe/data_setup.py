"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data(Food101).
"""
import os

from pathlib import Path

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()


def create_dataloaders(transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = num_workers):
    """Creates training and testing DataLoaders.

    Takes in a transform them and download food 101 dataset 
    and then into PyTorch DataLoaders.

    Args:
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
          train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(transform=some_transform,
                                 batch_size=32,
                                 num_workers=4)
    """
    # making dir for data
    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_data = torchvision.datasets.Food101(root=data_path,
                                              split="train",
                                              transform=transform,
                                              download=True)
    test_data = torchvision.datasets.Food101(root=data_path,
                                             split="test",
                                             transform=transform,
                                             download=True)
    # DataLoaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names
