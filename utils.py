"""
Shared utility functions.
"""

import torch
import numpy as np
from skimage.color import lab2rgb


def lab_to_rgb(L, ab):
    """
    Converts normalized L and ab tensors back into a viewable RGB image.

    Args:
        L: tensor of shape (1, H, W), normalized to [-1, 1]
        ab: tensor of shape (2, H, W), normalized to [-1, 1]

    Returns:
        numpy array (H, W, 3) RGB image in range [0, 1]
    """
    L = L * 50.0 + 50.0   # back to 0-100
    ab = ab * 128.0       # back to -128..127
    lab = torch.cat([L, ab], dim=0).permute(1, 2, 0).cpu().numpy()
    rgb = lab2rgb(lab.astype(np.float64))
    return rgb
