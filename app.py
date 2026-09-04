"""
Streamlit demo app for the image colorization model.

Usage:
    streamlit run app.py
"""

import torch
import numpy as np
import streamlit as st
from PIL import Image
from skimage.color import rgb2lab

import config
from models.unet import UNetColorizer
from utils import lab_to_rgb

st.set_page_config(page_title="AI Image Colorization", page_icon="🎨", layout="centered")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetColorizer(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/best_model.pth", map_location=device))
    model.eval()
    return model, device


def colorize_image(model, device, input_image):
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


st.title("🎨 AI Image Colorization")
st.write(
    "Upload any black & white image (landscape or portrait) and the model will "
    "automatically add color. Built with a U-Net (PyTorch), trained on a combined "
    "landscape + face dataset."
)

model, device = load_model()

uploaded_file = st.file_uploader("Upload a Black & White Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(input_image, use_container_width=True)

    with st.spinner("Colorizing..."):
        result_image = colorize_image(model, device, input_image)

    with col2:
        st.subheader("Colorized Result")
        st.image(result_image, use_container_width=True)
