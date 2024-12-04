import argparse
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from food_dataset import FoodDataset
from tqdm.auto import tqdm
import wandb


def save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, results_dir, best=False):
    """
    Save a checkpoint of the model.
    """
    if best:
        checkpoint_path = os.path.join(results_dir, "best_model.pth")
    else:
        checkpoint_path = os.path.join(results_dir, f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, checkpoint_path)
    if best:
        print(f"Best model checkpoint saved: {checkpoint_path}")
    else:
        print(f"Checkpoint saved: {checkpoint_path}")


def train(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    wandb.log({"Train Loss": avg_loss, "Epoch": epoch})
    return avg_loss


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Validating Epoch {epoch}/{total_epochs}", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy, "Epoch": epoch})
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-18 model on a food classification dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory of images.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--wandb_key", type=str, default="", help="wandb API Key for logging.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    # wandb initialization
    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key.strip())
        wandb.init(project="haochiai", config=vars(args))

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet-18 model
    model = models.resnet18()
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 208)  # Assuming 208 classes
    model = model.to(device)

    # Freeze layers except layer4 and fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = FoodDataset(csv_file=args.train_csv, root_dir=args.root_dir, transform=transform)
    val_dataset = FoodDataset(csv_file=args.val_csv, root_dir=args.root_dir, transform=transform)

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device, epoch, args.epochs)
        scheduler.step(val_loss)

        print(f"\nEpoch [{epoch}/{args.epochs}] "
              f"\nTrain Loss: {train_loss:.4f}, "
              f"\nVal Loss: {val_loss:.4f}, "
              f"\nVal Accuracy: {val_accuracy:.2f}%\n")

        # Save checkpoint every 10 epochs or when validation accuracy improves
        if (epoch % 10) == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, args.results_dir)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, args.results_dir, best=True)

        # Log metrics to wandb
        wandb.log({"Best Validation Accuracy": best_val_accuracy})


if __name__ == "__main__":
    main()
