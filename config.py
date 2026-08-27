"""
Central configuration file for the Image Colorization project.
Edit values here instead of hunting through scripts.
"""

import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---------- Image settings ----------
IMAGE_SIZE = 256          # images resized to 256x256
CHANNELS_L = 1             # grayscale (L channel)
CHANNELS_AB = 2            # color channels (a, b)

# ---------- Training settings ----------
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 50
LEARNING_RATE = 2e-4
BETA1 = 0.5                # Adam optimizer momentum (good default for GANs)
BETA2 = 0.999

# ---------- GAN specific ----------
LAMBDA_L1 = 100.0           # weight for L1 loss vs adversarial loss (Pix2Pix paper default)

# ---------- Device ----------
DEVICE = "cuda"  # will fall back to "cpu" automatically in code if unavailable

# ---------- Misc ----------
SEED = 42
SAVE_EVERY_N_EPOCHS = 5
