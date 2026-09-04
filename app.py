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

st.set_page_config(
    page_title="Image Colorization",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: linear-gradient(180deg, #fafafa 0%, #f2f4f7 100%);
        }

        .block-container {
            max-width: 880px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            margin-bottom: 1.75rem;
        }

        .hero h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem;
        }

        .hero p {
            color: #6b7280;
            font-size: 1.02rem;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.5;
        }

        [data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1.5px dashed #c7cbe8;
            border-radius: 14px;
            padding: 0.75rem;
            box-shadow: 0 2px 10px rgba(20, 20, 43, 0.04);
        }

        [data-testid="stFileUploader"] section {
            border: none;
        }

        .result-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1rem 1rem 1.25rem 1rem;
            box-shadow: 0 4px 18px rgba(20, 20, 43, 0.06);
            border: 1px solid #eef0f6;
        }

        .result-card h4 {
            margin: 0 0 0.6rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: #374151;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .result-card img {
            border-radius: 10px;
        }

        .footer-note {
            text-align: center;
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


st.markdown(
    """
    <div class="hero">
        <h1>🎨 Image Colorization</h1>
           <p>From grayscale to full color, in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, device = load_model()

uploaded_file = st.file_uploader(
    "Upload a black & white image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    with st.spinner("Colorizing your image..."):
        result_image = colorize_image(model, device, input_image)

    st.write("")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="result-card"><h4>Original</h4>', unsafe_allow_html=True)
        st.image(input_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card"><h4>Colorized</h4>', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    buf_col = st.columns([1, 1, 1])
    with buf_col[1]:
        import io
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        st.download_button(
            "Download result",
            data=buf.getvalue(),
            file_name="colorized.png",
            mime="image/png",
            use_container_width=True,
        )

st.markdown(
    '<div class="footer-note">Built with a U-Net (PyTorch) trained on a combined landscape + face dataset.</div>',
    unsafe_allow_html=True,
)
