"""
Gradio demo app for the image colorization model.

Usage:
    python app.py
"""

import torch
import numpy as np
import gradio as gr
from PIL import Image
from skimage.color import rgb2lab

import config
from models.unet import UNetColorizer
from utils import lab_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetColorizer(in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/best_model.pth", map_location=device))
model.eval()


def colorize_image(input_image):
    if input_image is None:
        return None

    img = input_image.convert("RGB")
    original_size = img.size

    img_resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    img_np = np.array(img_resized) / 255.0

    img_lab = rgb2lab(img_np).astype("float32")
    L = img_lab[:, :, 0] / 50.0 - 1.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        ab_pred = model(L_tensor).squeeze(0).cpu()

    L_for_convert = torch.from_numpy(L).unsqueeze(0)
    result_rgb = lab_to_rgb(L_for_convert, ab_pred)

    result_img = Image.fromarray((result_rgb * 255).astype(np.uint8))
    result_img = result_img.resize(original_size)

    return result_img


demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Upload a Black & White Image"),
    outputs=gr.Image(type="pil", label="Colorized Result"),
    title="AI Image Colorization",
    description=(
        "Upload any black & white image (landscape or portrait) and the model "
        "will automatically add color. Built with a U-Net (PyTorch), trained on "
        "a combined landscape + face dataset."
    ),
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
