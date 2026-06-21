import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import AdaINStyleTransferNet

# Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using local accelerator device: {DEVICE}")

IMAGE_SIZE = 512  
CHECKPOINT_PATH = "./checkpoints/decoder_epoch20.pth"
CONTENT_DIR = "./data/content"
STYLE_DIR = "./data/style"

content_files = [os.path.join(CONTENT_DIR, f) for f in os.listdir(CONTENT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
style_files = [os.path.join(STYLE_DIR, f) for f in os.listdir(STYLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

TEST_CONTENT = random.choice(content_files)
TEST_STYLE = random.choice(style_files)

print(f"Selected Random Content Image: {os.path.basename(TEST_CONTENT)}")
print(f"Selected Random Style Artwork: {os.path.basename(TEST_STYLE)}\n")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0) 
    return img_tensor.to(DEVICE)

def denormalize(tensor):
    img = tensor.cpu().clone().squeeze(0).numpy()
    img = img.transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img * std + mean
    return img.clip(0, 1)

# Load weights
print("Reassembling network modules...")
inference_net = AdaINStyleTransferNet().to(DEVICE)

# Extract decoder
target_decoder = inference_net.module.decoder if hasattr(inference_net, 'module') else inference_net.decoder

print(f"Loading trained weights from: {CHECKPOINT_PATH}")
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
target_decoder.load_state_dict(state_dict)
inference_net.eval()

# Load image tensors
content_tensor = load_image(TEST_CONTENT)
style_tensor = load_image(TEST_STYLE)

print("Running arbitrary style transfer optimization pass...")
with torch.no_grad():
    generated_tensor, _, _ = inference_net(content_tensor, style_tensor)

# Convert tensors back to viewable image arrays
content_display = denormalize(content_tensor)
style_display = denormalize(style_tensor)
generated_display = denormalize(generated_tensor)

# Plot Image
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(content_display)
axes[0].set_title("Original Content Image")
axes[0].axis("off")

axes[1].imshow(style_display)
axes[1].set_title("Target Style Artwork")
axes[1].axis("off")

axes[2].imshow(generated_display)
axes[2].set_title("Final Stylized Output Canvas")
axes[2].axis("off")

plt.tight_layout()

output_path = "./stylized_output.png"
plt.savefig(output_path, bbox_inches='tight')
print(f"Rendered canvas saved locally to: {output_path}")

plt.show()
print("Generation complete!")