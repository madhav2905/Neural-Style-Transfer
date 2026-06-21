import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# Encoder Layer
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained VGG19 feature extractor
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = vgg[0:2]    # relu1_1
        self.slice2 = vgg[2:7]    # relu2_1
        self.slice3 = vgg[7:12]   # relu3_1
        self.slice4 = vgg[12:21]  # relu4_1

        # Freeze VGG weights (For feature extraction)
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x):
        # Pass the input to extract feature maps 
        relu1_1 = self.slice1(x)
        relu2_1 = self.slice2(relu1_1)
        relu3_1 = self.slice3(relu2_1)
        relu4_1 = self.slice4(relu3_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1]
    
# Adain Layer
class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat):
        # Content mean and variance
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_var = content_feat.var(dim=[2, 3], keepdim=True) + self.eps
        content_std = content_var.sqrt()

        # Style mean and variance
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_var = style_feat.var(dim=[2, 3], keepdim=True) + self.eps
        style_std = style_var.sqrt()

        # Normalize content
        normalized = (content_feat - content_mean) / content_std 
        
        return normalized * style_std + style_mean
    
# Decoder Layer
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            # Mirror of slice4: 512 -> 256, upsample 32x32 -> 64x64
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),

            # Mirror of slice3: 256 -> 128, upsample 64x64 -> 128x128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),

            # Mirror of slice2: 128 -> 64, upsample 128x128 -> 256x256
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),

            # Mirror of slice1: 64 -> 3, final output
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3),
        )

    def forward(self, x):
        return self.decoder(x)
    
class AdaINStyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def forward(self, content_img, style_img):
        # Extract features from encoder
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)

        # Deep bottleneck features
        content_features_deep = content_features[-1]
        style_features_deep = style_features[-1]

        # Blend the statistics using AdaIN
        t = self.adain(content_features_deep, style_features_deep)  #t: Target Features 

        # Decode back to a standard 3-channel RGB image
        generated_img = self.decoder(t)

        return generated_img, style_features, t