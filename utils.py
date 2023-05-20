import torch
from pathlib import Path

import torch
from pathlib import Path


def save_model(model, optimizer, epoch, loss, directory, model_name='model', **kwargs):
    """
    Save a PyTorch model checkpoint.

    Args:
    model: Trained model.
    optimizer: Optimizer used for training.
    epoch: The last epoch the model was trained on.
    loss: The last loss recorded during training.
    directory: The directory where to save the model.
    model_name: Base name for the model file, defaults to 'model'.
    kwargs: Additional keyword arguments representing metrics to be included in the filename.
    To use the function, you would do something like this:
    >>>save_checkpoint(model, optimizer, epoch, loss, './model_dir', f1_score=val_f1score)
    """
    # Create the directory if it does not exist
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Create the filename
    metrics_str = '_'.join(f'{key}={value:.4f}' for key, value in kwargs.items())
    filename = f'{directory}/{model_name}_epoch={epoch}_loss={loss:.4f}_{metrics_str}.pth'

    # Save the model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }, filename)


def get_device() -> torch.device:
    """
        Retrieves the appropriate Torch device for running computations.

        Returns:
            torch.device: The Torch device to be used for computations.

        Raises:
            None

        Examples:
            >>> device = get_device()
            >>> print(device)
            cuda

        """
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple GPU
    else:
        device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
    # print(f"Using {device} device")
    return torch.device(device)


def load_checkpoint(model, optimizer, filename):
    """
    Load a PyTorch model checkpoint.

    Args:
    model: Model to load the weights into.
    optimizer: Optimizer to load the state into.
    filename: The path of the checkpoint file.

    Returns:
    The epoch at which training was stopped, the last loss recorded, and any additional metrics.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Extract additional metrics
    metrics = {key: value for key, value in checkpoint.items() if
               key not in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']}

    return epoch, loss, metrics

# To use the function, you would do something like this:
# epoch, loss, metrics = load_checkpoint(model, optimizer, 'model_checkpoint.pth')
