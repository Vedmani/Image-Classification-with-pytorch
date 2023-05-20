import torch
from pathlib import Path
from models import EfficientNet


class CheckpointCallback:
    def __init__(self, checkpoint_dir, monitor='val_loss', mode='min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_epoch = None
        self.best_model_path = None

    def __call__(self, epoch, model, optimizer, **kwargs):
        # Calculate the value of the monitored metric
        monitor_value = kwargs.get(self.monitor)
        if monitor_value is None:
            raise ValueError(f"monitor value for metric {self.monitor} is not available.")

        # Check if the current epoch has a better score than the previous best score
        if self.best_score is None or \
                (self.mode == 'min' and monitor_value < self.best_score) or \
                (self.mode == 'max' and monitor_value > self.best_score):
            # Save the current model checkpoint
            self.best_score = monitor_value
            self.best_epoch = epoch
            self.best_model_path = self.checkpoint_dir / f"best_model_{epoch}.pt"

            state_dict = {'epoch': epoch,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'best_score': self.best_score,
                          'best_epoch': self.best_epoch}

            torch.save(state_dict, self.best_model_path)

            print(f"Saved checkpoint for epoch {epoch} with {self.monitor}={monitor_value:.4f}")

    def load_checkpoint(self, checkpoint_path, device, args):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model = EfficientNet()
        model = torch.compile(model)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])

        self.best_score = checkpoint['best_score']
        self.best_epoch = checkpoint['best_epoch']

        return epoch + 1, model, optimizer