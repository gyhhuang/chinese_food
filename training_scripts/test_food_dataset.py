import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TestFoodDataset(Dataset):
    def __init__(self, metadata_file, base_path, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.base_path, row['image_path'])
        label = row['dish_id']

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
