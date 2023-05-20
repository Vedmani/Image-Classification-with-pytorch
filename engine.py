import torch
from typing import Callable


def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    loss_fn,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    device: torch.device,
                    f1score: Callable,
                    accuracy: Callable,
                    DEBUG: bool = False) -> tuple:
    """
    Trains the model for one epoch using the specified dataloader and returns the average training loss,
    accuracy and F1-score for the epoch.

    Args:
        model: The neural network model to be trained
        epoch: The current epoch number
        train_dataloader: DataLoader object containing the training data
        loss_fn: The loss function to be used for optimization
        optimizer: The optimization algorithm to be used
        scheduler: The learning rate scheduler to be used
        device: The device on which the computations should be carried out
        f1score: The F1-score metric to be used for evaluation
        accuracy: The accuracy metric to be used for evaluation
        DEBUG: A boolean flag to enable debug mode

    Returns:
        Tuple[float, float, float]: A tuple containing the average training loss, accuracy and F1-score for the epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize the running totals for the loss, accuracy and F1-score.
    train_acc_total, train_loss_total, train_f1score_total = 0, 0, 0
    # Iterate over the training dataloader.
    for batch_idx, (X, y) in enumerate(train_dataloader):
        # If DEBUG is enabled, break after the first 10 batches.
        if DEBUG and batch_idx == 10:
            break
        # Move the data to the device.
        X, y = X.to(device), y.to(device)
        # Forward pass.
        y_pred = model(X)
        # Calculate the loss.
        loss = loss_fn(y_pred, y)
        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        # Update the learning rate scheduler.
        scheduler.step()
        # Calculate the accuracy and F1-score.
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        f1score_metrics = f1score(y_pred_class, y)
        acc = accuracy(y_pred_class, y)
        # Print a progress message.
        msg = f"Batch: {batch_idx + 1} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f} | F1Score: {f1score_metrics:.4f}"
        print('\r' + msg, end='', flush=True)
        # Update the running totals.
        train_loss_total = train_loss_total + loss.item()
        train_acc_total = train_acc_total + acc
        train_f1score_total = train_f1score_total + f1score_metrics
    # Calculate the average loss, accuracy and F1-score.
    train_loss_avg = train_loss_total / len(train_dataloader)
    train_acc_avg = train_acc_total / len(train_dataloader)
    train_f1score_avg = train_f1score_total / len(train_dataloader)
    # If DEBUG is enabled, calculate the average loss, accuracy and F1-score for the first 10 batches.
    if DEBUG:
        train_loss_avg = train_loss_total / 10
        train_acc_avg = train_acc_total / 10
        train_f1score_avg = train_f1score_total / 10
    # Print the epoch history.
    epoch_history = f"Loss: {train_loss_avg:.4f} | Accuracy: {train_acc_avg:.4f} | F1Score: {train_f1score_avg:.4f}"
    print('\r' + epoch_history, end='\n', flush=True)
    return train_loss_avg, train_acc_avg, train_f1score_avg


# function to validate the model for one epoch
def validate_one_epoch(model: torch.nn.Module,
                       val_dataloader: torch.utils.data.DataLoader,
                       loss_fn,
                       device: torch.device,
                       f1score: Callable,
                       accuracy: Callable,
                       DEBUG: bool = False) -> tuple:
    """
        Function to validate the model for one epoch.

        Parameters:
        model: The model to validate.
        val_dataloader: Dataloader for the validation set.
        loss_fn: Loss function used for validation.
        device: The device type used (GPU or CPU).
        f1score: Function to calculate the F1 score.
        accuracy: Function to calculate the accuracy.
        DEBUG: A boolean flag used for debugging. If true, only 10 batches will be validated.

        Returns:
        A tuple containing average validation loss, accuracy, and F1 score for the epoch.
    """
    # Switch model to evaluation mode
    model.eval()
    # Initialize counters for total validation loss, accuracy and F1 score
    val_acc_total, val_loss_total, val_f1score_total = 0, 0, 0
    # Disabling gradient calculation as we are in validation mode
    with torch.no_grad():
        # Loop through all batches in the validation dataloader
        for batch_idx, (X, y) in enumerate(val_dataloader):
            # Debug condition
            if DEBUG and batch_idx == 10:
                break
            # Move the batch tensors to the same device as the model
            X, y = X.to(device), y.to(device)
            # Forward pass
            y_pred = model(X)
            # Calculate the loss
            loss = loss_fn(y_pred, y)
            # Calculate the accuracy and F1-score
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            f1score_metrics = f1score(y_pred_class, y)
            acc = accuracy(y_pred_class, y)
            # Update the running totals
            val_loss_total = val_loss_total + loss.item()
            val_acc_total = val_acc_total + acc
            val_f1score_total = val_f1score_total + f1score_metrics
            # Print a progress message
            msg = f"Batch: {batch_idx + 1} | Validation Loss: {loss.item():.4f} | Validation Accuracy: {acc:.4f} | Validation F1Score: {f1score_metrics:.4f}"
            print('\r' + msg, end='', flush=True)
        # Calculate the average loss, accuracy and F1-score
        val_loss_avg = val_loss_total / len(val_dataloader)
        val_acc_avg = val_acc_total / len(val_dataloader)
        val_f1score_avg = val_f1score_total / len(val_dataloader)
        # If DEBUG is enabled, calculate the average loss, accuracy and F1-score for the first 10 batches.
        if DEBUG:
            val_loss_avg = val_loss_total / 10
            val_acc_avg = val_acc_total / 10
            val_f1score_avg = val_f1score_total / 10
        epoch_history = f"Validation Loss: {val_loss_avg:.4f} | Validation Accuracy: {val_acc_avg:.4f} | Validation F1Score: {val_f1score_avg:.4f}"
        print('\r' + epoch_history, end='\n', flush=True)
    return val_loss_avg, val_acc_avg, val_f1score_avg
