from engine import train_one_epoch, validate_one_epoch
import torch
from dataloaders import get_train_val_dataloader
from models import EfficientNet
from torchmetrics import Accuracy, F1Score
from config import get_args
import time
from utils import save_model
import wandb
from tqdm import tqdm
from discordutils import send_msg
from wandb_utils import log_values, initialize_wandb_run


def main(args):
    DEBUG = args.DEBUG
    if DEBUG:
        print("Debugging mode is on. Only 10 batches will be used for training and validation.")
    loss = torch.nn.CrossEntropyLoss()
    f1score = F1Score(task="multiclass", num_classes=25, average="macro").to(args.device)
    accuracy = Accuracy(task='multiclass', num_classes=25).to(args.device)
    model = EfficientNet()
    # model = torch.compile(model)
    train_dataloader, _, _ = get_train_val_dataloader(root_dir=args.data_dir,
                                                                         batch_size=args.batch_size)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=len(
        train_dataloader), epochs=args.epochs)
    logs = train(args, args.device, loss, f1score, accuracy, model, optimizer, scheduler=scheduler)
    print(logs)


def train(args, device, loss, f1score, accuracy, model, optimizer, starting_epoch=0, scheduler=None):
    if args.wandb:
        run = initialize_wandb_run(args)
    print("Using train-test split.")
    DEBUG = args.DEBUG
    train_loss_list, train_acc_list, train_f1score_list = [], [], []
    val_loss_list, val_acc_list, val_f1score_list = [], [], []
    train_dataloader, val_dataloader, classes = get_train_val_dataloader(root_dir=args.data_dir,
                                                                         batch_size=args.batch_size)
    if args.wandb:
        run.watch(model, log="gradients", log_freq=100)
    for epoch in tqdm(range(starting_epoch, args.epochs + starting_epoch), disable=False, desc="Epochs"):
        start_time = time.time()
        print(f"\nEpoch: {epoch}")
        epoch_loss, epoch_acc, epoch_f1score = train_one_epoch(model, train_dataloader, loss, optimizer, scheduler, device,
                                                               f1score, accuracy, DEBUG)
        epoch_val_loss, epoch_val_acc, epoch_val_f1score = validate_one_epoch(model, val_dataloader, loss, device,
                                                                              f1score, accuracy, DEBUG)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch time: {epoch_time:.2f} seconds")
        if args.wandb:
            log_values(run, step=epoch, loss=epoch_loss, f1score=epoch_f1score, accuracy=epoch_acc, val_loss=epoch_val_loss, val_f1score=epoch_val_f1score, val_accuracy=epoch_val_acc)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)
        train_f1score_list.append(epoch_f1score)
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)
        val_f1score_list.append(epoch_val_f1score)
        # run.alert(title="Epoch completed", text=f"Epoch {epoch + 1} completed with F1score: {epoch_val_f1score:.4f}.")
        if args.discord:
            send_msg(f"Epoch {epoch} completed with F1score: {epoch_val_f1score:.4f}.")
        save_model(model_name="efficientnet",
                   model=model,
                   optimizer=optimizer,
                   epoch=epoch,
                   loss=epoch_loss,
                   val_f1score=epoch_val_f1score,
                   directory=args.model_dir)
    if args.wandb:
        run.finish()
    if args.discord:
        send_msg("Training complete")
    return train_loss_list, train_acc_list, train_f1score_list, val_loss_list, val_acc_list, val_f1score_list


if __name__ == "__main__":
    args = get_args()
    main(args)