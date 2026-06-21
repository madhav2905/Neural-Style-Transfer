import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import StyleTransferDataset
from model import AdaINStyleTransferNet


# Loss helper functions
def calc_mean_std(feat, eps=1e-5):
    """Calculates channel-wise mean and standard deviation for style loss"""
    mean = feat.mean(dim=[2, 3], keepdim=True)
    std = (feat.var(dim=[2, 3], keepdim=True) + eps).sqrt()
    return mean, std


def compute_content_loss(gen_feat, t):
    return F.mse_loss(gen_feat, t)


def compute_style_loss(gen_feats, style_feats):
    loss = 0
    for gen_feat, style_feat in zip(gen_feats, style_feats):
        gen_mean, gen_std = calc_mean_std(gen_feat)
        style_mean, style_std = calc_mean_std(style_feat)
        loss += F.mse_loss(gen_mean, style_mean) + F.mse_loss(gen_std, style_std)
    return loss


# Config
CONTENT_DIR = "data/content"
STYLE_DIR = "data/style"
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
LAMBDA_STYLE = 10.0
LOG_EVERY = 100
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY_EPOCH = True

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Setup
device = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
print("Using device:", device)

model = AdaINStyleTransferNet().to(device)

dataset = StyleTransferDataset(CONTENT_DIR, STYLE_DIR, image_size=256)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

optimizer = torch.optim.Adam(model.decoder.parameters(), lr=LR)


# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (content, style) in enumerate(dataloader):
        content = content.to(device)
        style = style.to(device)

        # Forward propagation
        generated, style_feats, t = model(content, style)

        # Re-extract features of the generated canvas via frozen encoder layers
        gen_feats = model.encoder(generated)

        # Run loss evaluations
        c_loss = compute_content_loss(gen_feats[-1], t)
        s_loss = compute_style_loss(gen_feats, style_feats)

        total_loss = c_loss + LAMBDA_STYLE * s_loss

        # Backpropagation weight adjustments
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log updates to terminal display
        if i % LOG_EVERY == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Iter [{i}/{len(dataloader)}] "
                  f"Content Loss: {c_loss.item():.4f} "
                  f"Style Loss: {s_loss.item():.4f} "
                  f"Total Loss: {total_loss.item():.4f}")

    if SAVE_EVERY_EPOCH:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"decoder_epoch{epoch+1}.pth")
        torch.save(model.decoder.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

print("Training complete.")