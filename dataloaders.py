import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


def get_train_val_dataloader(root_dir,
                             batch_size=32,
                             num_workers=os.cpu_count()):
    """
        Get the train and validation dataloaders for a given dataset.

        Args:
            root_dir: The path to the dataset directory.
            batch_size: The batch size.
            num_workers: The number of workers to use for data loading.

        Returns:
            The train and validation dataloaders.
        """
    data_transform = transforms.Compose([
        transforms.Resize(size=(150, 150)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = ImageFolder(root_dir, transform=data_transform)
    train_indices, val_indices = train_test_split(
        list(range(len(data))),
        test_size=0.2,
        stratify=data.targets)
    # Create the train and validation datasets
    train_dataset = torch.utils.data.Subset(data, train_indices)
    val_dataset = torch.utils.data.Subset(data, val_indices)
    # Create the train and validation dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Number of batches in train_dataloader: {len(train_dataloader)}")
    print(f"Number of batches in val_dataloader: {len(val_dataloader)}")
    print(f"Train: {len(train_indices)} samples")
    print(f"Validation: {len(val_indices)} samples")

    return train_dataloader, val_dataloader, data.class_to_idx


def get_kfold_loaders(root_dir, num_splits=5, batch_size=32, num_workers=os.cpu_count(), shuffle=True):
    """
        Generates train and validation data loaders for each fold in Stratified Group K-Fold cross-validation.

        Args:
            root_dir (str): Root directory of the dataset.
            num_splits (int, optional): Number of folds for cross-validation.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            num_workers (int, optional): Number of worker threads for data loaders. Defaults to the number of CPU cores.
            shuffle (bool, optional): Whether to shuffle the data during training. Defaults to True.

        Yields:
            tuple: Tuple containing train and validation data loaders for each fold.

        """
    print(f"Number of CPU cores: {num_workers}")
    # Write transform for image
    data_transform = transforms.Compose([
        transforms.Resize(size=(150, 150)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root_dir, transform=data_transform)

    # Get the file paths and labels of all images in the datasets
    all_image_paths = [sample[0] for sample in dataset.samples]
    all_image_labels = [sample[1] for sample in dataset.samples]

    # Define the number of splits for StratifiedGroupKFold
    n_splits = 5

    # Initialize the StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X=all_image_paths, y=all_image_labels, groups=all_image_paths)):
        # Create train and validation datasets based on the indices from k-fold split
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        # Create train and validation data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      sampler=None, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    sampler=None, pin_memory=True)
        # print number of batches in each dataloader
        if fold == 0:
            print(f"Number of batches in train_dataloader: {len(train_dataloader)}")
            print(f"Number of batches in val_dataloader: {len(val_dataloader)}")
            print(f"Train: {len(train_idx)} samples")
            print(f"Validation: {len(val_idx)} samples")
        yield train_dataloader, val_dataloader
    return dataset.class_to_idx
