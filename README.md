# Image-classification-with-pytorch

This repository contains code for training a bird classification model using PyTorch and demonstrating the model using a Gradio app. The model is trained on a dataset of 25 different bird classes.

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

This section will contain instructions for installing the required libraries. (TODO)

## Weights & Biases Setup

Weights & Biases (wandb) is a tool for tracking and visualizing machine learning experiments. This section will contain instructions for setting up wandb. (TODO)

## Discord Webhook Setup

Discord webhook is a simple way to automate sending messages to your Discord channels. This section will contain instructions for setting up a Discord webhook. (TODO)

## Usage

To start training, use the following command:
```bash
python main.py --data_dir /path/to/data --model_dir /path/to/model --epochs 10 --batch_size 32 --learning_rate 0.01
```
Replace /path/to/data and /path/to/model with the paths to your data and model directories, respectively. Adjust the other arguments as needed.
To run the Gradio app, use the following command:
```bash
python app.py
```
## Model Training
TODO
## Tools and References
TODO
## Project Status and Future Plans
This project is currently undergoing active development. Your valuable feedback and suggestions are highly appreciated. I will continuously update the repository with detailed instructions on utilizing this code to train your personalized image classifier model. Moreover, as I continue to expand my knowledge, I will consistently incorporate additional features into this repository.
