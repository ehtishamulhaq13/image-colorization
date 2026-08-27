"""
Training script for the image colorization model.

Usage:
    python train.py

Expects data already split into data/train, data/val, data/test
(see notebooks/ or README for how the dataset was prepared).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ColorizationDataset
from models.unet import UNetColorizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for L, ab in tqdm(loader, desc="Training"):
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        pred_ab = model(L)
        loss = criterion(pred_ab, ab)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for L, ab in tqdm(loader, desc="Validating"):
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)
            running_loss += loss.item()
    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_dataset = ColorizationDataset(config.TRAIN_DIR, image_size=config.IMAGE_SIZE)
    val_dataset = ColorizationDataset(config.VAL_DIR, image_size=config.IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = UNetColorizer(in_channels=1, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    print(f"Starting training for {config.EPOCHS} epochs...\n")
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"  Best model saved! (val_loss: {val_loss:.4f})")

        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            torch.save(
                model.state_dict(),
                os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth"),
            )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
