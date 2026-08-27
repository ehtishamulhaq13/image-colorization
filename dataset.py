"""
PyTorch Dataset for the colorization task.
Converts RGB images to L*a*b* color space: L channel is the input,
a/b channels are the target the model learns to predict.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab
import numpy as np


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        img_np = np.array(img) / 255.0
        img_lab = rgb2lab(img_np).astype("float32")

        # L: 0-100 -> normalized to -1..1
        L = img_lab[:, :, 0] / 50.0 - 1.0
        # ab: -128..127 -> normalized to -1..1
        ab = img_lab[:, :, 1:] / 128.0

        L = torch.from_numpy(L).unsqueeze(0).float()          # (1, H, W)
        ab = torch.from_numpy(ab).permute(2, 0, 1).float()    # (2, H, W)

        return L, ab
