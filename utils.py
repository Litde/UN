import json
import os
from torchviz import make_dot
import torch

EXPECTED_KEYS = {"COMET_API_KEY", "COMET_PROJECT_NAME", "COMET_WORKSPACE_NAME"}

def load_comet_credentials(path: str) -> dict:
    """
    Load and validate a credentials JSON file with fields:
    {
        "COMET_API_KEY": "",
        "COMET_PROJECT_NAME": "",
        "COMET_WORKSPACE_NAME": ""
    }

    Returns:
        dict: The parsed credentials.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or keys are missing/extra.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Credentials file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not valid JSON: {e}")

    data_keys = set(data.keys())

    missing = EXPECTED_KEYS - data_keys
    extra = data_keys - EXPECTED_KEYS

    if missing:
        raise ValueError(f"Credentials file is missing keys: {missing}")
    if extra:
        raise ValueError(f"Credentials file contains unexpected keys: {extra}")

    for key in EXPECTED_KEYS:
        if not isinstance(data[key], str):
            raise ValueError(f"Key '{key}' must be a string, got {type(data[key]).__name__}")

    return data


from torchviz import make_dot
import torch


def plot_unet_architecture(model, input_shape=(1, 3, 256, 256), filename="architecture/unet_architecture"):
    """
    Generates an architecture plot for the UNet model using torchviz.

    Args:
        model: PyTorch model (nn.Module)
        input_shape: Shape of a dummy input tensor
        filename: Output file name (without extension)
    """
    model.eval()

    # Make dummy input
    x = torch.randn(input_shape)

    # Forward pass
    y = model(x)

    # Create graph
    graph = make_dot(y, params=dict(model.named_parameters()))

    # Save diagram (PDF + PNG)
    graph.render(filename, format="png")
    graph.render(filename, format="pdf")

    print(f"Architecture saved as: {filename}.png and {filename}.pdf")




