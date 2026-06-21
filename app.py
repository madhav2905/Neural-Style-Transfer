import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from model import AdaINStyleTransferNet

# Configuration & Layout
st.set_page_config(
    page_title="Neural Style Transfer Canvas",
    page_icon="🎨",
    layout="wide"
)

st.title("Neural Style Transfer Generator ✨")
st.write("Upload a content image and a style artwork to blend them together!")

@st.cache_resource
def get_device_and_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    net = AdaINStyleTransferNet().to(device)
    target_decoder = net.module.decoder if hasattr(net, 'module') else net.decoder
    
    checkpoint_path = "./checkpoints/decoder_epoch20.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "decoder_epoch20.pth"
        
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        target_decoder.load_state_dict(state_dict)
    else:
        st.error(f"Model weights file not found! Please ensure 'decoder_epoch20.pth' is accessible.")
        
    net.eval()
    return device, net

device, inference_net = get_device_and_model()

# Data preprocessing
IMAGE_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_inference(content_image, style_image):
    c_tensor = transform(content_image.convert("RGB")).unsqueeze(0).to(device)
    s_tensor = transform(style_image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated_tensor, _, _ = inference_net(c_tensor, s_tensor)
        
    img = generated_tensor.cpu().clone().squeeze(0).numpy().transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img * std + mean
    img = img.clip(0, 1)
    return img

# UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Content Image")
    content_file = st.file_uploader("Choose a background image...", type=["jpg", "jpeg", "png"], key="content")
    if content_file:
        st.image(content_file, caption="Target Content Image Structure", use_container_width=True)

with col2:
    st.subheader("2. Style Image")
    style_file = st.file_uploader("Choose an artwork pattern...", type=["jpg", "jpeg", "png"], key="style")
    if style_file:
        st.image(style_file, caption="Target Style Texture Base", use_container_width=True)

st.markdown("---")

# Execution & Generation
if content_file and style_file:
    if st.button("Render Masterpiece Canvas", use_container_width=True):
        c_img = Image.open(content_file)
        s_img = Image.open(style_file)
        
        with st.spinner("Parsing layer statistics and painting features... Please hold."):
            output_array = run_inference(c_img, s_img)
            
        st.success("Style Transfer Rendering Complete!")

        _, out_col, _ = st.columns([1, 2, 1])
        with out_col:
            st.subheader("Your Generated Artwork")
            st.image(output_array, caption="Processed Feed-Forward Composition Output", use_container_width=True)
else:
    st.info("Complete steps 1 and 2 by uploading images above to unlock the generation pass engine.")