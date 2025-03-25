
import warnings

from sklearn.metrics import average_precision_score
from  model.model import BiTNet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
from model.utils import calculate_map_segmentation_per_batch, plot_losses,calculate_map_iou_thresholds_per_batch,plot_map50,plot_map5095
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    print (sys.path)
    # Set device
    print(f"Using device: {device}")
    batch_size = 4
    num_workers = 2
    num_epochs = 1
    learning_rate = 1e-4
    weight_decay = 1e-5
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = torchvision.datasets.VOCSegmentation(
        root="./VOC2012data",
        year="2012",
    image_set="train",
    download=True,  # Assuming already downloaded
    transform=transform,
    target_transform=transform
    )

    val_dataset = torchvision.datasets.VOCSegmentation(
        root="./VOC2012data",
        year="2012",
        image_set="val",
        download=True,
        transform=transform,
        target_transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Initialize model
    model = BiTNet(num_classes=21)  # VOC has 20 classes + background
    model = model.to(device)
    num_classes=21
# Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
        # Hyperparameters



# Initialize lists to store losses
    train_losses = []
    val_losses = []
    map50_history=[]
    map5095_history=[]
    train_map_history = []
    val_map_history = []
# Training loop
    for epoch in range(num_epochs):
    # Training phase
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Move data to device
            images = images.to(device)
            masks = masks.to(device)

        # VOC masks come as (B, 1, H, W) - squeeze channel dimension
            masks = masks.squeeze(1).long()

        # Forward pass
            outputs = model(images)

        # Compute loss
            loss = criterion(outputs, masks)

        # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_map = calculate_map_segmentation_per_batch(model, train_loader, device, num_classes)
        train_map_history.append(train_map)
    # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                masks = masks.to(device)
                masks = masks.squeeze(1).long()

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

    # Calculate average validation loss for this epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        map50, map5095, ap_per_class = calculate_map_iou_thresholds_per_batch(
        model, val_loader, device, num_classes=5, return_per_class=True
        )
        map50_history.append(map50)
        map5095_history.append(map5095)
    # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"mAP@50: {map50:.4f}",
                f"mAP@50:95:{map5095:.4f}")

    # Plot losses after each epoch
    plot_losses(train_losses, val_losses)
    plot_map50(map50_history)
    plot_map5095(map5095_history)
def cuda():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))




if __name__ == "__main__":
    main()