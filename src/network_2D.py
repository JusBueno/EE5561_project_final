import torch
import torch.nn as nn
import numpy as np
from typing import Literal
import math

class VAE_UNET_2D_M01(nn.Module):
    def __init__(self, in_channels, input_dim=np.asarray([192,128], dtype=np.int64), num_classes = 4, VAE_enable=True, HR_layers=0, fusion: Literal["None", "Slab", "Modality", "Hybrid"] = "None"):
        super(VAE_UNET_2D_M01, self).__init__()

        self.VAE_enable = VAE_enable
        self.HR_layers = HR_layers
        
        self.old_in_shape = tuple(input_dim // (2 ** HR_layers))
        self.shape_trans, self.new_in_shape = make_divisible_crop_or_pad(self.old_in_shape, 16)
        self.input_shape = np.asarray([self.new_in_shape[0], self.new_in_shape[1]], dtype=np.int64)
        self.shape_inv_trans = make_crop_or_pad(self.new_in_shape, tuple(input_dim))

        # Dimensions
        self.input_dim = input_dim # Dimensions of a slice
        self.enc_dim = self.input_shape // 8 # Encoder Output Dimension
        self.VAE_C1 = np.floor((self.enc_dim - 1) / 2) + 1

        # Encoder Layers
        self.E1 = nn.Sequential(
            FusionLayer(in_channels // 4, fusion=fusion),
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
        if self.HR_layers == 0:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv2d(32, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.HR_layers == 1:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.D5 = nn.Sequential(
                ResidualBlock(16),
                nn.Conv2d(16, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.HR_layers == 2:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.D5 = nn.Sequential(
                ResidualBlock(16),
                nn.Conv2d(16, 8, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.D6 = nn.Sequential(
                ResidualBlock(8),
                nn.Conv2d(8, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise TypeError("Invalid Number of HR Layers")
        
        if self.VAE_enable:
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

            if self.HR_layers == 0:
                self.Vend = nn.Conv2d(32, in_channels, kernel_size=1)
            elif self.HR_layers == 1:
                self.VUpHR0 = nn.Sequential(
                    nn.Conv2d(32, 16, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR0 = nn.Sequential(
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1)
                )
                self.Vend = nn.Conv2d(16, in_channels, kernel_size=1)
            elif self.HR_layers == 2:
                self.VUpHR1 = nn.Sequential(
                    nn.Conv2d(32, 16, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR1 = nn.Sequential(
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1)
                )
                self.VUpHR0 = nn.Sequential(
                    nn.Conv2d(16, 8, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR0 = nn.Sequential(
                    nn.GroupNorm(8, 8),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 8),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1)
                )
                self.Vend = nn.Conv2d(8, in_channels, kernel_size=1)
            else:
                raise TypeError("Invalid Number of HR Layers")
        
    def init_weights_gaussian(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        

    def forward(self, x):
        # Encoder layers
        x_trans = self.shape_trans(x)
        enc_out_1 = self.E1(x_trans)
        enc_out_2 = self.E2(enc_out_1)
        enc_out_3 = self.E3(enc_out_2)
        enc_out_4 = self.E4(enc_out_3)

        # Decoder layers
        dec_out = self.D1(enc_out_4)
        dec_out = self.D2(enc_out_3 + dec_out)
        dec_out = self.D3(enc_out_2 + dec_out)
        
        if self.HR_layers == 0:
            dec_out = self.D4(enc_out_1 + dec_out)
        elif self.HR_layers == 1:
            dec_out = self.D4(enc_out_1 + dec_out)
            dec_out = self.D5(dec_out)
        elif self.HR_layers == 2:
            dec_out = self.D4(enc_out_1 + dec_out)
            dec_out = self.D5(dec_out)
            dec_out = self.D6(dec_out)
        else:
            raise TypeError("Invalid Number of HR Layers")

        dec_out = self.shape_inv_trans(dec_out)
        

        if self.VAE_enable:
            # VAE layers
            VAE_out = self.VD(enc_out_4)
            distr = VAE_out
            #mu, logvar = torch.chunk(VAE_out, 2, dim=1)  # split into two 128-sized vectors
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

            if self.HR_layers == 0:
                VAE_out = self.Vend(VAE_out)
            elif self.HR_layers == 1:
                VAE_out = self.VUpHR0(VAE_out)
                VAE_out = VAE_out + self.VBlockHR0(VAE_out)
                VAE_out = self.Vend(VAE_out)
            elif self.HR_layers == 2:
                VAE_out = self.VUpHR1(VAE_out)
                VAE_out = VAE_out + self.VBlockHR1(VAE_out)
                VAE_out = self.VUpHR0(VAE_out)
                VAE_out = VAE_out + self.VBlockHR0(VAE_out)
                VAE_out = self.Vend(VAE_out)
            else:
                raise TypeError("Invalid Number of HR Layers")
            VAE_out = self.shape_inv_trans(VAE_out)
            return dec_out, VAE_out, distr
        else:
            return dec_out, 0, 0


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
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.res_layers(x)


class FusionLayer(nn.Module):
    """
    Fusion layer for multi-modal 2D slabs.

    Input:  (B, modalities*N, H, W), modality-first
    Output: (B, out_channels, H, W)

    Fusion modes:
        "None"      -> standard conv
        "Slab"      -> fuse slabs within each modality, preserve modalities
        "Modality"  -> fuse modalities per slab, preserve slabs
        "Hybrid"    -> combines Slab and Modality fusion together 
    """
    def __init__(self, N: int, 
                 fusion: Literal["None", "Slab", "Modality", "Hybrid"] = "None",
                 out_channels: int = 32,
                 modalities: int = 4):
        super().__init__()

        valid = ["None", "Slab", "Modality", "Hybrid"]
        
        if fusion not in valid:
            raise ValueError(f"Invalid fusion mode '{fusion}'. Expected one of {valid}.")

        self.N = N
        self.fusion = fusion
        self.out_channels = out_channels
        self.modalities = modalities
        
        if fusion == "None":
            self.out_layer = nn.Conv2d(self.modalities * self.N, self.out_channels, kernel_size=3, padding=1)
        
        elif fusion == "Slab":
            # Handle slabs to be first convolved in groups,
            # Meaning that modality is preserved
            # Group size of modalities (4)
            if self.out_channels % self.modalities != 0:
                raise ValueError("out_channels must be divisible by modalities for slab fusion")
            self.out_layer = nn.Conv2d(self.modalities * self.N, self.out_channels, kernel_size=3, padding=1, groups=self.modalities)
  
        elif fusion == "Modality":
            # Handle modalities to be first convolved in groups,
            # Meaning that slabs are preserved
            # Group size N
            # The input needs to be reorganized to slab first
            out_channels_ceil = math.ceil(self.out_channels / N) * N
            self.out_layer = nn.Sequential(
                nn.Conv2d(self.modalities * self.N, out_channels_ceil, kernel_size=1, groups=self.N),
                nn.Conv2d(out_channels_ceil, self.out_channels, kernel_size=3, padding=1)
            )

        elif fusion == "Hybrid":
            # Combines slab and modality fusion

            # Slab fuser
            self.slab_channels = self.out_channels // 2
            if self.slab_channels % self.modalities != 0:
                raise ValueError("out_channels // 2 must be divisible by modalities for slab fusion")
            self.slab_fuser = nn.Conv2d(self.modalities * self.N, self.slab_channels, kernel_size=3, padding=1, groups=self.modalities)
            
            # Modality fuser
            out_channels_ceil = math.ceil((self.out_channels // 2) / N) * N
            self.mod_fuser = nn.Sequential(
                nn.Conv2d(self.modalities * self.N, out_channels_ceil, kernel_size=1, groups=self.N),
                nn.Conv2d(out_channels_ceil, self.out_channels // 2, kernel_size=3, padding=1)
            )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 4N, H, W) modality-first
        """
        B, C, H, W = x.shape
        N = self.N
        M = C // N

        # Sanity check
        if self.modalities * self.N != C:
            raise ValueError(f"Invalid input size, expected {self.modalities * self.N} channels, but got {C}")
        
        if self.fusion == "Hybrid":
            # Slab fusion
            slab_out = self.slab_fuser(x)

            # Reorganize to slab first for modality fusion
            x_mod = x.view(B,M,N,H,W)
            x_mod  = x_mod.permute(0,2,1,3,4)
            x_mod = x_mod.reshape(B, M*N, H, W)
            mod_out = self.mod_fuser(x_mod)
            
            # Return combined fusions
            return torch.cat([slab_out, mod_out], dim=1)
        
        if self.fusion == "Modality":
            # Reorganize to slab first
            x_mod = x.view(B,M,N,H,W)
            x_mod  = x_mod.permute(0,2,1,3,4)
            x_mod = x_mod.reshape(B, M*N, H, W)
            return self.out_layer(x_mod)
        
        else:
            return self.out_layer(x)


# Handling undivisible sizes (2D version)
def make_divisible_crop_or_pad(shape, s):
    """
    shape: tuple (H, W)
    s: scalar (usually power of 2)

    Returns:
      transform: function(tensor) that crops/pads last 2 dims to multiples of s
      (H2, W2): divisible-by-s output spatial size
    """
    import torch.nn.functional as F

    H, W = shape

    def target_size(x):
        return int(((x + s - 1) // s) * s)

    H2 = target_size(H)
    W2 = target_size(W)

    def transform(tensor):
        """
        tensor shape: (..., H, W)
        returns:      (..., H2, W2)
        """
        h, w = tensor.shape[-2:]

        t = tensor

        # ---- CROP ----
        if h > H2:
            t = t[..., :H2, :]
        if w > W2:
            t = t[..., :, :W2]

        # ---- PAD ----
        pad_h = max(0, H2 - t.shape[-2])
        pad_w = max(0, W2 - t.shape[-1])

        # F.pad order for 2D: (w_before, w_after, h_before, h_after)
        padding = (0, pad_w, 0, pad_h)
        t = F.pad(t, padding)

        return t

    return transform, (H2, W2)


def make_crop_or_pad(in_shape, out_shape):
    """
    2D version
    in_shape:  (H, W)
    out_shape: (H2, W2)

    Returns: transform(tensor) → (..., H2, W2)
    """
    import torch.nn.functional as F

    assert len(in_shape) == 2 and len(out_shape) == 2, "2D only (H, W)"

    in_dims  = list(in_shape)
    out_dims = list(out_shape)
    D = 2

    def transform(tensor):
        """
        tensor: (..., H, W)
        """
        t = tensor

        # ---- CROP ----
        for i in range(D):
            cur = t.shape[-D + i]
            target = out_dims[i]

            if cur > target:
                slices = [slice(None)] * t.ndim
                slices[-D + i] = slice(0, target)
                t = t[tuple(slices)]

        # ---- PAD ----
        pads = []
        # F.pad: (w_before, w_after, h_before, h_after)
        for i in reversed(range(D)):  # W then H
            cur = t.shape[-D + i]
            target = out_dims[i]
            pad_amount = max(0, target - cur)
            pads.extend([0, pad_amount])

        if any(pads):
            t = F.pad(t, pads)

        return t

    return transform


# Local Test
if __name__ == "__main__":
    inChans = 4 * 3 # 4 Modalities, 3 Slices
    VAE_enable = True
    fusion = input("Fusion Options (None, Slab, Modality, Hybrid): ")
    if fusion not in ["None", "Slab", "Modality", "Hybrid"]:
        fusion = "None"
        print("Invalid Fusion, defaulting to None")

    print("Local Test - 2D Reference Net - Our Implementation")
    print("\nReference Network - Done by us - MOD 01")
    print("Asymmetric Network")

    print("\nWith VAE")
    
    print("\n0 HR")
    model_VAE_0HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=0, fusion=fusion)
    
    x = torch.randn(1, inChans, 240, 240) 
    out1, out2, out3 = model_VAE_0HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n1 HR")
    model_VAE_1HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=1)
    
    x = torch.randn(1, inChans, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_VAE_1HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n2 HR")
    model_VAE_2HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=2)
    
    x = torch.randn(1, inChans, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_VAE_2HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\nNo VAE")
    print("\n0 HR")
    model_noVAE_0HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=0)
    
    x = torch.randn(1, inChans, 240, 240) 
    out1, out2, out3 = model_noVAE_0HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n1 HR")
    model_noVAE_1HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=1)
    
    x = torch.randn(1, inChans, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_noVAE_1HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n2 HR")
    model_noVAE_2HR = VAE_UNET_2D_M01(in_channels=inChans, input_dim=np.asarray([240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=2)
    
    x = torch.randn(1, inChans, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_noVAE_2HR(x)

    print("Output 1 shape:", out1.shape)

    