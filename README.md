# Image-classification-with-pytorch

This repository contains code for training (finetuning) a image classification model using PyTorch and demonstrating the model using a Gradio app. The model is trained on a dataset of 25 different bird classes.

## Files in the Repository

- `app.py`: This file contains a Gradio app for demonstrating the bird classification model. It includes functions to load the model, transform the input image, and predict the class of the image. The Gradio interface is created and launched in this file.

- `config.py`: This file contains the configuration for the application. It uses the argparse module to parse command-line arguments. The arguments include:
    - `data_dir`: The directory where the data is stored.
    - `model_dir`: The directory where the model will be saved.
    - `run_id`: The unique identifier for the run.
    - `epochs`: The number of epochs for training.
    - `batch_size`: The batch size for training.
    - `learning_rate`: The learning rate for the optimizer.
    - `DEBUG`: Whether to run in debug mode. If this argument is used, it will be set to True.
    - `wandb`: Whether to use Weights & Biases for logging. If this argument is used, it will be set to True.
    - `discord`: Whether to send notifications to Discord. If this argument is used, it will be set to True.

- `dataloaders.py`: This file contains the data loaders for the application. It uses the torchvision module to load and preprocess the data. It includes functions to create the training and validation data loaders.

- `discordutils.py`: This file contains utility functions for sending notifications to Discord. It includes functions to create a Discord webhook and to send a message to Discord.

- `engine.py`: This file contains the engine for the application. It defines the training and evaluation loops. It includes functions to perform one epoch of training or evaluation, and to calculate the accuracy of the predictions.

- `idx_to_class.json`: This file contains a mapping from class indices to class names. It is used to convert the output of the model into a human-readable class name.

- `main.py`: This file is the entry point for the application. It parses command-line arguments, initializes the data loaders and the model, and starts the training process.

- `models.py`: This file contains the model definitions. It uses the timm module to create an EfficientNet model. It includes a function to create the model and a function to count the number of parameters in the model. The number of output features of the last layer of the model should be changed according to the number of classes in the dataset.

- `utils.py`: This file contains utility functions for saving and loading model checkpoints, and for getting the device for computations. It includes functions to save a checkpoint, to load a checkpoint, and to get the device.

- `wandb_utils.py`: This file contains utility functions for initializing a Weights & Biases run and for logging values to Weights & Biases. It includes functions to initialize a Weights & Biases run and to log values to Weights & Biases.

## Dataset

The model is trained on a dataset of 25 different bird classes. You can download the dataset from [here](https://www.kaggle.com/datasets/ichhadhari/indian-birds).

## Requirements

To install the required libraries, you can use pip, a package installer for Python. You just need to run the following command in your terminal:

```sh
pip install -r requirements.txt
```

## Weights & Biases Setup

Weights & Biases (wandb) is a tool for tracking and visualizing machine learning experiments. To set up wandb, follow these steps:

1. Install the wandb library using pip:

```sh
pip install wandb
```

2. Sign up for a free account on the [Weights & Biases website](https://wandb.ai/site).

3. Run `wandb login` in your terminal and follow the instructions.

## Discord Webhook Setup

Discord webhook is a simple way to automate sending messages to your Discord channels. To set up a Discord webhook, follow these steps:

1. Go to the settings of the Discord channel you want to send messages to.

2. Click on 'Integrations', then 'Webhooks', and finally 'New Webhook'.

3. Copy the Webhook URL and use it in [`discordutils.py`]("discordutils.py").

## Model Training

The model is trained using the `train_one_epoch` function from [`engine.py`](engine.py). This function takes in the model, dataloader, loss function, optimizer, scheduler, device, and metrics as arguments. It performs one epoch of training, calculates the loss, accuracy, and F1-score for each batch, and returns the average loss, accuracy, and F1-score for the epoch.

The training process can be started by running the [`main.py`](main.py) script with the appropriate command-line arguments. For example:

```sh
python main.py --data_dir /path/to/data --model_dir /path/to/model --epochs 10 --batch_size 32 --learning_rate 0.01
```

## Tools and References

This project uses the following tools and libraries:

- [PyTorch](https://pytorch.org/): An open-source machine learning library for Python, used for building and training the model.
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models): A collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

- [Weights & Biases](https://wandb.ai/site): A tool for tracking and visualizing machine learning experiments.

- [Discord Webhook](https://discord.com/developers/docs/resources/webhook): A simple way to automate sending messages to Discord channels.

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946): Mingxing Tan, Quoc V. Le. ICML 2019.

For more information on how to use these tools, please refer to their official documentation.

## Project Status and Future Plans

This project is currently undergoing active development. Your valuable feedback and suggestions are highly appreciated. I will continuously update the repository with detailed instructions on utilizing this code to train your personalized image classifier model. Moreover, as I continue to expand my knowledge, I will consistently incorporate additional features into this repository. Future plans include adding more models, improving the training process, and providing more detailed documentation.