from models import EfficientNet
from utils import get_device
import torch
import json
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import json
import timm
from torch import nn
import torch.nn.functional as F

def load_efficientnet_model(model_path: str, device=get_device()):
    """
    Load a PyTorch model checkpoint.

    Args:
        model_path: The path of the checkpoint file.
        device: The device to load the model onto.

    Returns:
        The model loaded onto the specified device.
    """
    # Initialize model
    model = EfficientNet()

    # Load model weights onto the specified device
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    return model

with open('idx_to_class.json', 'r') as f:
    idx_to_class = json.load(f)


def predict_image(array):
    """
    Predict the class of an image.

    Args:
        array: The image data as an array.

    Returns:
        The predicted class.
    """
    # Convert the image to a PIL Image object
    input_image = Image.fromarray(array)

    # Load the model
    model = load_efficientnet_model('efficientnet_epoch=18_loss=0.0020_val_f1score=0.8993.pth')

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize(size=(150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = transform(input_image).unsqueeze(0)
    image.to(get_device())

    # Predict the class
    with torch.no_grad():
        output = model(image)
        # Apply softmax to the outputs to convert them into probabilities
        probabilities = F.softmax(output, dim=1)
        predicted = probabilities.argmax().item()
        predicted_class = idx_to_class[str(predicted)]  # Make sure your keys in json are string type

    return predicted_class


# Create the image classifier
image_classifier = gr.Interface(fn=predict_image, inputs="image", outputs="text")

# Launch the image classifier
image_classifier.launch(share=True)
