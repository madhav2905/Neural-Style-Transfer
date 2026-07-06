# Neural Style Transfer — AdaIN
A real-time, feed-forward Neural Style Transfer (NST) application built with **PyTorch** and deployed with a clean **Streamlit** interface. This project utilizes an arbitrary style transfer network leveraging **Adaptive Instance Normalization (AdaIN)**, optimized to run completely on a free cloud CPU tier.

Upload any content photo and any style artwork — the model transfers the artistic style onto the content image in a single forward pass, no iterative optimization required.

**[Live Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/madhav-0209/neural-style-transfer)**

## Project Overview

Traditional neural style transfer methods (Gatys et al., 2015) rely on an optimization process that requires hundreds of backpropagation loops per image, making real-time CPU deployment impossible. 

This implementation separates the heavy lifting into an offline training phase on a dual-GPU cluster. During deployment, the network performs a **single feed-forward pass**. By shifting content features to match the mean and variance statistics of the style features via AdaIN, the model renders $512 \times 512$ masterpieces in under **3 to 5 seconds on a standard cloud CPU**.

### Key Features
* **Arbitrary Style Transfer:** Blends any custom content image with any custom artistic texture map dynamically.
* **Highly Optimized Footprint:** The entire trained decoder weights file is squeezed down, keeping memory thrashing low and container cold-starts near-instantaneous.
* **Robust Production Setup:** Pre-configured with custom server-side CORS handling to allow seamless embedded iframe data streaming on cloud platforms.

## How It Works
 
Classic Neural Style Transfer produces high-quality results but is slow — it runs gradient descent for hundreds of iterations per image. AdaIN solves this by transferring style in **feature space** rather than pixel space, enabling real-time inference.
 
The key insight: the *mean* and *standard deviation* of CNN feature maps encode the visual style of an image. AdaIN normalizes the content image's features and re-scales them using the style image's statistics:
 
```
AdaIN(x, y) = σ(y) · ((x − μ(x)) / σ(x)) + μ(y)
```
 
where `x` = content features, `y` = style features, computed per-channel over spatial dimensions.

## Dataset
 
| Split | Dataset | Images | Purpose |
|---|---|---|---|
| Content | MS-COCO val2017 | 5,000 | Teaches decoder to reconstruct natural scenes |
| Style | WikiArt | 8,500 | Exposes model to diverse artistic textures and palettes |
 
No paired data or labels required — content and style images are randomly sampled independently during training.

## Architecture
1. **Feature Extraction:** A pre-trained VGG-19 network slices out feature representations for both content and style inputs.
2. **AdaIN Layer:** The content feature maps are normalized and scaled to match the exact statistical distributions (mean and variance) of the style feature maps.
3. **Decoder Network:** A custom symmetric decoder mirrors the VGG layers, reconstructing the stylized target feature maps back into a beautifully rendered RGB image canvas.
 
```
Content Image ──┐
                ├──► VGG-19 Encoder (frozen) ──► AdaIN ──► Decoder ──► Stylized Output
Style Image   ──┘
```

**VGG-19 Encoder (frozen)**
- Pretrained on ImageNet, used purely as a fixed feature extractor
- Extracts features at 4 layers: `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`
- Only `relu4_1` features fed into AdaIN; all 4 used for style loss

**AdaIN Layer**
- Normalizes content features (zero mean, unit variance per channel)
- Re-scales using style image's per-channel mean and std
- No learnable parameters — pure statistical transfer

**Decoder (trained from scratch)**
- Mirrors the encoder architecture in reverse
- Uses `ReflectionPad2d` + `Conv2d` + `Upsample (nearest)` instead of MaxPool
- Only trained component in the entire pipeline

## Project Structure
 
```
Neural-Style-Transfer/
├── dataset.py          # PyTorch Dataset — loads COCO + WikiArt, applies transforms
├── model.py            # VGGEncoder, AdaIN, Decoder, AdaINStyleTransferNet
├── train.py            # Training loop with content + style loss
├── test_inference.py   # Local inference script 
├── test_dataset.py     # Dataset validation script
├── app.py              # Streamlit web app
├── model.ipynb         # Jupyter notebook for model development
├── outputs/            # Sample stylized outputs
├── requirements.txt
└── .gitignore
```

## Local Setup
 
**1. Clone the repo**
```bash
git clone https://github.com/madhav2905/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```
 
**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
**3. Download trained weights**
 
Download `decoder_epoch20.pth` from the [releases page](#) or train your own (see below), and place it in a `checkpoints/` folder:
```
checkpoints/
└── decoder_epoch20.pth
```
 
**4. Run inference**
```bash
python test_inference.py
```
 
Randomly picks a content + style image from `data/content/` and `data/style/`, runs inference, and saves the result to `stylized_output.png`.

## Deployment
 
The app is deployed as a **Streamlit** web application on **Hugging Face Spaces**.
 
**Try it live:** [huggingface.co/spaces/madhav-0209/neural-style-transfer](https://huggingface.co/spaces/madhav-0209/neural-style-transfer)
 
To run the app locally:
```bash
pip install streamlit
streamlit run app.py
```

## License
MIT License
