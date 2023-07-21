"""
Build a EfficientNet B2 model from torchvision
"""

import torch
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_build(device=device):
    # getting the weights for EfficientNet B2 and then get the transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT  # ".DEFAULT" = best available weights
    # Transforms used in EfficientNet B2
    transform = weights.transforms()
    # Model
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    # print(model)
    return model
