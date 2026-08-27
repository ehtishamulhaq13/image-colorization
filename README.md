# 🎨 AI Image Colorization (Landscape + Portrait)

Deep learning project that automatically adds color to black & white images using a U-Net architecture built with PyTorch. Trained on a combined dataset of landscape photography and human faces, so it generalizes across both scene types.

## 🧠 How It Works

Instead of predicting RGB values directly, the model works in **L\*a\*b\* color space**:
- **L channel** (lightness/grayscale) → used as the model's **input**
- **a, b channels** (color information) → predicted by the model as **output**

This separation lets the network focus purely on learning color, since the structural/brightness information is already given.

## 🏗️ Architecture

- **Model:** U-Net (encoder-decoder with skip connections)
- **Input:** 1-channel grayscale image (256×256)
- **Output:** 2-channel color prediction (a, b channels, 256×256)
- **Parameters:** ~54.4M
- **Loss function:** L1 Loss
- **Optimizer:** Adam (lr=2e-4, betas=(0.5, 0.999))

## 📊 Dataset

- **Landscapes:** [Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) (Kaggle) — 4,319 images
- **Faces:** [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (Kaggle) — 3,000 sampled images
- **Total:** 7,319 images
- **Split:** 80% train / 10% validation / 10% test

## 💾 Pre-trained Model

The trained model weights (`best_model.pth`, ~200MB) are too large for GitHub. Download them here:

**[Download best_model.pth from Google Drive](https://drive.google.com/file/d/1DMZMw9TDLOzmowSZ-WzSF2NgXk5zFYfC/view?usp=sharing)**

After downloading, place the file inside the `checkpoints/` folder before running `app.py`.

## 🎯 Results

Model trained for 40 epochs on a combined landscape + portrait dataset. It successfully colorizes both natural scenes (sky, grass, sand) and human portraits (skin tone, hair color).

*Sample results available in `outputs/`*

**Known limitation:** Performs best on images similar in style to the training data (natural lighting, standard photography). Stylized/editorial photography with unusual lighting may produce more muted results.

## 🚀 Getting Started

### Requirements
```bash
pip install -r requirements.txt
```

### Run the Demo App
```bash
python app.py
```
This launches a Gradio interface where you can upload any black & white image and get a colorized result.

### Train From Scratch
See `train.py` for the full training pipeline, or follow the step-by-step notebook in `notebooks/`.

## 📁 Project Structure
```
image-colorization/
├── config.py              # Central configuration (hyperparameters, paths)
├── requirements.txt        # Dependencies
├── models/
│   └── unet.py             # U-Net architecture
├── data/                   # Dataset (train/val/test splits)
├── checkpoints/             # Saved model weights
├── outputs/                 # Sample results
├── app.py                   # Gradio demo app
├── train.py                 # Training script
└── README.md
```

## 🔮 Future Improvements
- GAN-based approach (Pix2Pix) for more vibrant, realistic colors
- Larger/more diverse training dataset
- Perceptual loss (VGG-based) for more natural color transitions

## 📝 License
This project is open source and available for educational purposes.
