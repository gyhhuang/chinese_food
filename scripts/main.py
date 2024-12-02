import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from food_dataset import FoodDataset
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 208)
    model = model.to(device)

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True 
        else:
            param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root_dir = "images/all/"
    train_path = "parsed_data/train_data.csv"
    val_path = "parsed_data/test_data.csv"

    # Create datasets
    train_dataset = FoodDataset(csv_file=train_path, root_dir=root_dir, transform=transform)
    val_dataset = FoodDataset(csv_file=val_path, root_dir=root_dir, transform=transform)

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
