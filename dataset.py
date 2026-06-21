import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_dir = content_dir
        self.style_dir = style_dir

        self.content_images = [
            os.path.join(content_dir, img)
            for img in os.listdir(content_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.style_images = [
            os.path.join(style_dir, img)
            for img in os.listdir(style_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize(280),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return max(len(self.content_images), len(self.style_images))

    def __getitem__(self, idx):
        content_path = self.content_images[idx % len(self.content_images)]
        style_path = random.choice(self.style_images)

        content_image = Image.open(content_path).convert("RGB")
        style_image = Image.open(style_path).convert("RGB")

        content_image = self.transform(content_image)
        style_image = self.transform(style_image)

        return content_image, style_image