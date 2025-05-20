# satellite_classifier.py
# Main script for EuroSAT satellite image classification using EfficientNet

# Install required packages
pip install torchinfo torchviz

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset configuration
data_dir = r'EuroSAT_RGB'
if os.path.exists(data_dir):
    print(f"Dataset found at: {data_dir}")
else:
    raise FileNotFoundError(f"Dataset not found at: {data_dir}")

# Image transformations
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                         saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load and split dataset
dataset = ImageFolder(root=data_dir, transform=base_transform)
print(f"Total samples: {len(dataset)}")

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Apply augmentation only to training set
train_dataset.dataset.transform = train_transform

# Create data loaders
batch_size = 64
num_workers = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

print("Data loaders created successfully")