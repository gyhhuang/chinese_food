import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


def get_transforms():
    """
    Returns the image transformations to apply to the dataset.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_resnet_model(model_path, num_classes):
    """
    Loads a pretrained ResNet-50 model and adjusts it for the number of classes.
    """
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
