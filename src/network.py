import torch
import torch.nn as nn
import numpy as np


# Variational Autoencoder-Regularized U-Net
# Assuming use of a single modality
# Takes slices as channels
# Assuming full image (160x192x128)
class VAE_UNET(nn.Module):
    def __init__(self, in_channels, input_dim=np.asarray([192,128], dtype=np.int64), HR_dim=np.asarray([192,128], dtype=np.int64), num_classes = 4):
        super(VAE_UNET, self).__init__()

        # Dimensions
        self.input_dim = input_dim # Dimensions of a slice
        self.enc_dim = self.input_dim // 8 # Encoder Output Dimension
        self.VAE_C1 = np.floor((self.enc_dim - 1) / 2) + 1
        
        # Encoder Layers
        self.E1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2), 
            ResidualBlock(32)
        )
        self.E2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.E3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.E4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder Layers
        self.D1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.D2 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.D3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        # Modified for HR output
        self.D4 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, 32, kernel_size=1), # Keeping the same number of filters
            nn.Upsample(size=(HR_dim[0], HR_dim[1]), mode='bilinear')
        )

        self.D5 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, num_classes-1, kernel_size=1),
            nn.Sigmoid()
        )

        # VAE Layers
        self.VD = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(int(16 * self.VAE_C1[0] * self.VAE_C1[1]), 256)
        ) 

        self.VDraw = GaussianSample()

        # VU layer is split to add the reshaping in the middle
        self.VU_1 = nn.Sequential(
            nn.Linear(128, int(256 * (self.enc_dim[0] // 2) * (self.enc_dim[1] // 2))),
            nn.ReLU()
        )
        self.VU_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.VUp2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        # Do not forget to add input after this
        self.VBlock2 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.VUp1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        # Do not forget to add input after this
        self.VBlock1 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.VUp0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        # Do not forget to add input after this
        self.VBlock0 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )

        # Custom layer to handle superresolution
        self.VUp_HR = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Upsample(size=(HR_dim[0], HR_dim[1]), mode='bilinear')
        )

        # Do not forget to add input after this
        self.VBlock_HR = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )

        self.Vend = nn.Conv2d(32, 1, kernel_size=1)
        
    def init_weights_gaussian(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        

    def forward(self, x):
        # Encoder layers
        enc_out_1 = self.E1(x)
        enc_out_2 = self.E2(enc_out_1)
        enc_out_3 = self.E3(enc_out_2)
        enc_out_4 = self.E4(enc_out_3)

        # Decoder layers
        dec_out = self.D1(enc_out_4)
        dec_out = self.D2(enc_out_3 + dec_out)
        dec_out = self.D3(enc_out_2 + dec_out)
        dec_out = self.D4(enc_out_1 + dec_out)
        dec_out = self.D5(dec_out)

        # VAE layers
        VAE_out = self.VD(enc_out_4)
        mu, logvar = torch.chunk(VAE_out, 2, dim=1)  # split into two 128-sized vectors
        VAE_out = self.VDraw(VAE_out)
        VAE_out = self.VU_1(VAE_out)
        VAE_out = VAE_out.view(-1, 256, self.enc_dim[0] // 2, self.enc_dim[1] // 2)
        VAE_out = self.VU_2(VAE_out)
        VAE_out = self.VUp2(VAE_out)
        VAE_out = VAE_out + self.VBlock2(VAE_out)
        VAE_out = self.VUp1(VAE_out)
        VAE_out = VAE_out + self.VBlock1(VAE_out)
        VAE_out = self.VUp0(VAE_out)
        VAE_out = VAE_out + self.VBlock0(VAE_out)
        VAE_out = self.VUp_HR(VAE_out)
        VAE_out = VAE_out + self.VBlock_HR(VAE_out)
        VAE_out = self.Vend(VAE_out)

        return dec_out, VAE_out, mu, logvar

    
# Class for Gaussian Distribution Sample
class GaussianSample(nn.Module):
    def forward(self, x):
        # x shape: (B, 256)
        mu, logvar = torch.chunk(x, 2, dim=1)  # split into two 128-sized vectors
        std = torch.exp(0.5 * logvar)          # convert logvar to std
        eps = torch.randn_like(std)            # sample ε ~ N(0, I)
        z = mu + eps * std                     # reparameterization
        return z
    
# Class for Residual Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_layers = nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(in_channels, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.res_layers(x)

# Simple UNET based on paper
class UNET(nn.Module):
    def __init__(self, in_channels, num_classes = 4):
        super(UNET, self).__init__()

        # Encoder Layers
        self.E1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2), 
            ResidualBlock(32)
        )
        self.E2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.E3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.E4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder Layers
        self.D1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.D2 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.D3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.D4 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, num_classes-1, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Encoder layers
        enc_out_1 = self.E1(x)
        enc_out_2 = self.E2(enc_out_1)
        enc_out_3 = self.E3(enc_out_2)
        enc_out_4 = self.E4(enc_out_3)

        # Decoder layers
        dec_out = self.D1(enc_out_4)
        dec_out = self.D2(enc_out_3 + dec_out)
        dec_out = self.D3(enc_out_2 + dec_out)
        dec_out = self.D4(enc_out_1 + dec_out)

        return dec_out


"""
HR_input = np.asarray([192, 128], np.int64)
LR_input = HR_input / 2
test_net = VAE_UNET(3, LR_input, HR_input)
x = torch.rand(4, 3, int(LR_input[0]), int(LR_input[1]))
D, V = test_net(x)
print("Input shape:", x.shape)
print("Decoder Output shape:", D.shape)
print("VAE output shape:", V.shape)
"""

if __name__ == "__main__":
    print("Testing Network")
    test_net = UNET(1, 4)
    x = torch.rand(4, 1, 240, 240)
    D = test_net(x)
    print("Input shape:", x.shape)
    print("Decoder Output shape:", D.shape)
