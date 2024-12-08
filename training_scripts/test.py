import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from log_food_item import load_food_data
from food_dataset import TestFoodDataset

# ANSI Escape Codes for color-coding results
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def get_paths(base_path):
    """
    Returns the paths for the dataset, images, and model based on the base path.
    """
    true_labels_csv = "../data_scripts/csv/food_dict_final.csv"
    metadata_path = os.path.join(base_path, "real_data.csv")
    images_dir = os.path.join(base_path, "images")
    model_path = os.path.join(base_path, "best_model.pth")
    return true_labels_csv, metadata_path, images_dir, model_path


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
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def test_model(model, test_loader, food_data_csv, device):
    """
    Tests the model on the given test dataset.
    """
    correct_count = 0
    total_samples = len(test_loader.dataset)

    print("Testing selected samples...")
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        predicted = torch.max(output, 1)[1].item()
        correct = label.item() == predicted

        # Extracting true and predicted food English names
        true_food_name = food_data_csv.get(str(label.item()), {}).get("english", "Unknown")
        predicted_food_name = food_data_csv.get(str(predicted), {}).get("english", "Unknown")

        # ANSI coloring for correct/incorrect predictions
        color = GREEN if correct else RED
        result_text = (
            f"Expected: {label.item()} ({true_food_name}), Predicted: {predicted} ({predicted_food_name})"
        )
        print(color + result_text + RESET)

        if correct:
            correct_count += 1

    accuracy = (correct_count / total_samples) * 100
    print(f'\nTest accuracy: {accuracy:.2f}%')
    return accuracy


def main():
    base_path = "../data_scripts/rw_test_set"
    true_labels_csv, metadata_path, images_dir, model_path = get_paths(base_path)

    # Prepare dataset and data loader
    transform = get_transforms()
    test_dataset = TestFoodDataset(metadata_path, images_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Load true labels and model
    food_data_csv = load_food_data(true_labels_csv)
    model = load_resnet_model(model_path, num_classes=208)

    # Set device and run testing
    device = torch.device("cpu")
    model.to(device)

    test_accuracy = test_model(model, test_loader, food_data_csv, device)


if __name__ == "__main__":
    main()
