import argparse
import os
from utils import get_device


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, default="data",
                        help="path to data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")

    # Model parameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="path to checkpoints")

    # Debug parameters
    parser.add_argument("--DEBUG", action="store_true", default=False,
                        help="debug mode")

    # Wandb parameters
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="use wandb")
    parser.add_argument("--run_id", type=str, default=None,
                        help="wandb run id")

    #discord
    parser.add_argument("--discord", action="store_true", default=False,
                        help="use discord")

    args = parser.parse_args()

    # If DEBUG is used, set wandb to false
    if args.DEBUG:
        args.wandb = False

    # Device parameter
    args.device = get_device()

    return args
