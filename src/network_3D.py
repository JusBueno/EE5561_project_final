import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class REF_VAE_UNET_3D(nn.Module):
    def __init__(self, in_channels, input_dim=np.asarray([160,192,128], dtype=np.int64), num_classes = 4, VAE_enable=True):
        super(REF_VAE_UNET_3D, self).__init__()

        self.VAE_enable = VAE_enable

        # Dimensions
        self.input_dim = input_dim # Dimensions of a slice
        self.enc_dim = self.input_dim // 8 # Encoder Output Dimension
        self.VAE_C1 = np.floor((self.enc_dim - 1) / 2) + 1
        
        # Encoder Layers
        self.E1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.Dropout3d(p=0.2), 
            ResidualBlock(32)
        )
        self.E2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.E3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.E4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder Layers
        self.D1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D2 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )

        self.D4 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv3d(32, num_classes-1, kernel_size=1),
            nn.Sigmoid()
        )
        
        if self.VAE_enable:
            # VAE Layers
            self.VD = nn.Sequential(
                nn.GroupNorm(8, 256),
                nn.ReLU(),
                nn.Conv3d(256, 16, kernel_size=3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(int(16 * self.VAE_C1[0] * self.VAE_C1[1] * self.VAE_C1[2]), 256)
            ) 

            self.VDraw = GaussianSample()

            # VU layer is split to add the reshaping in the middle
            self.VU_1 = nn.Sequential(
                nn.Linear(128, int(256 * (self.enc_dim[0] // 2) * (self.enc_dim[1] // 2) * (self.enc_dim[2] // 2))),
                nn.ReLU()
            )
            self.VU_2 = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            self.VUp2 = nn.Sequential(
                nn.Conv3d(256, 128, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock2 = nn.Sequential(
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1)
            )

            self.VUp1 = nn.Sequential(
                nn.Conv3d(128, 64, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock1 = nn.Sequential(
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1)
            )

            self.VUp0 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock0 = nn.Sequential(
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1)
            )

            self.Vend = nn.Conv3d(32, in_channels, kernel_size=1)
        
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

        if self.VAE_enable:
            # VAE layers
            VAE_out = self.VD(enc_out_4)
            distr = VAE_out
            #mu, logvar = torch.chunk(VAE_out, 2, dim=1)  # split into two 128-sized vectors
            VAE_out = self.VDraw(VAE_out)
            VAE_out = self.VU_1(VAE_out)
            VAE_out = VAE_out.view(-1, 256, self.enc_dim[0] // 2, self.enc_dim[1] // 2, self.enc_dim[1] // 2)
            VAE_out = self.VU_2(VAE_out)
            VAE_out = self.VUp2(VAE_out)
            VAE_out = VAE_out + self.VBlock2(VAE_out)
            VAE_out = self.VUp1(VAE_out)
            VAE_out = VAE_out + self.VBlock1(VAE_out)
            VAE_out = self.VUp0(VAE_out)
            VAE_out = VAE_out + self.VBlock0(VAE_out)
            VAE_out = self.Vend(VAE_out)

            return dec_out, VAE_out, distr
        else:
            return dec_out, 0, 0

class VAE_UNET_3D_M01(nn.Module):
    def __init__(self, in_channels, input_dim=np.asarray([160,192,128], dtype=np.int64), num_classes = 4, VAE_enable=True, HR_layers=0):
        super(VAE_UNET_3D_M01, self).__init__()

        self.VAE_enable = VAE_enable
        self.HR_layers = HR_layers
        
        self.old_in_shape = tuple(input_dim // (2 ** HR_layers))
        self.shape_trans, self.new_in_shape = make_divisible_crop_or_pad(self.old_in_shape, 16)
        self.input_shape = np.asarray([self.new_in_shape[0], self.new_in_shape[1], self.new_in_shape[2]], dtype=np.int64)
        self.shape_inv_trans = make_crop_or_pad(self.new_in_shape, tuple(input_dim))

        # Dimensions
        self.input_dim = input_dim # Dimensions of a slice
        self.enc_dim = self.input_shape // 8 # Encoder Output Dimension
        self.VAE_C1 = np.floor((self.enc_dim - 1) / 2) + 1

        # Encoder Layers
        self.E1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.Dropout3d(p=0.2), 
            ResidualBlock(32)
        )
        self.E2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.E3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.E4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder Layers
        self.D1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D2 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        if self.HR_layers == 0:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv3d(32, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.HR_layers == 1:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv3d(32, 16, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            self.D5 = nn.Sequential(
                ResidualBlock(16),
                nn.Conv3d(16, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.HR_layers == 2:
            self.D4 = nn.Sequential(
                ResidualBlock(32),
                nn.Conv3d(32, 16, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            self.D5 = nn.Sequential(
                ResidualBlock(16),
                nn.Conv3d(16, 8, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )
            self.D6 = nn.Sequential(
                ResidualBlock(8),
                nn.Conv3d(8, num_classes-1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise TypeError("Invalid Number of HR Layers")
        
        if self.VAE_enable:
            # VAE Layers
            self.VD = nn.Sequential(
                nn.GroupNorm(8, 256),
                nn.ReLU(),
                nn.Conv3d(256, 16, kernel_size=3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(int(16 * self.VAE_C1[0] * self.VAE_C1[1] * self.VAE_C1[2]), 256)
            ) 

            self.VDraw = GaussianSample()

            # VU layer is split to add the reshaping in the middle
            self.VU_1 = nn.Sequential(
                nn.Linear(128, int(256 * (self.enc_dim[0] // 2) * (self.enc_dim[1] // 2) * (self.enc_dim[2] // 2))),
                nn.ReLU()
            )
            self.VU_2 = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            self.VUp2 = nn.Sequential(
                nn.Conv3d(256, 128, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock2 = nn.Sequential(
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1)
            )

            self.VUp1 = nn.Sequential(
                nn.Conv3d(128, 64, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock1 = nn.Sequential(
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1)
            )

            self.VUp0 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock0 = nn.Sequential(
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1)
            )

            if self.HR_layers == 0:
                self.Vend = nn.Conv3d(32, in_channels, kernel_size=1)
            elif self.HR_layers == 1:
                self.VUpHR0 = nn.Sequential(
                    nn.Conv3d(32, 16, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR0 = nn.Sequential(
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv3d(16, 16, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv3d(16, 16, kernel_size=3, padding=1)
                )
                self.Vend = nn.Conv3d(16, in_channels, kernel_size=1)
            elif self.HR_layers == 2:
                self.VUpHR1 = nn.Sequential(
                    nn.Conv3d(32, 16, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR1 = nn.Sequential(
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv3d(16, 16, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 16),
                    nn.ReLU(),
                    nn.Conv3d(16, 16, kernel_size=3, padding=1)
                )
                self.VUpHR0 = nn.Sequential(
                    nn.Conv3d(16, 8, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                )
                # Do not forget to add input after this
                self.VBlockHR0 = nn.Sequential(
                    nn.GroupNorm(8, 8),
                    nn.ReLU(),
                    nn.Conv3d(8, 8, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 8),
                    nn.ReLU(),
                    nn.Conv3d(8, 8, kernel_size=3, padding=1)
                )
                self.Vend = nn.Conv3d(8, in_channels, kernel_size=1)
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
            VAE_out = VAE_out.view(-1, 256, self.enc_dim[0] // 2, self.enc_dim[1] // 2, self.enc_dim[2] // 2)
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

class VAE_UNET_3D_M04(nn.Module):
    def __init__(self, in_channels, input_dim=np.asarray([160,192,128], dtype=np.int64), num_classes = 4, VAE_enable=True, HR_layers=0):
        super(VAE_UNET_3D_M04, self).__init__()

        self.VAE_enable = VAE_enable
        self.HR_layers = HR_layers

        # Dimensions
        self.input_dim = input_dim # Dimensions of a slice
        self.enc_dim = self.input_dim // 8 # Encoder Output Dimension
        self.VAE_C1 = np.floor((self.enc_dim - 1) / 2) + 1
        
        # Encoder Layers
        self.E1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.Dropout3d(p=0.2), 
            ResidualBlock(32)
        )
        self.E2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2 if HR_layers == 0 else 1, padding=1),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.E3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2 if HR_layers <= 1 else 1, padding=1),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.E4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder Layers
        self.D1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D2 = nn.Sequential(
            ResidualBlock(128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.D3 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )

        self.D4 = nn.Sequential(
            ResidualBlock(32),
            nn.Conv3d(32, num_classes-1, kernel_size=1),
            nn.Sigmoid()
        )
        
        if self.VAE_enable:
            # VAE Layers
            self.VD = nn.Sequential(
                nn.GroupNorm(8, 256),
                nn.ReLU(),
                nn.Conv3d(256, 16, kernel_size=3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(int(16 * self.VAE_C1[0] * self.VAE_C1[1] * self.VAE_C1[2]), 256)
            ) 

            self.VDraw = GaussianSample()

            # VU layer is split to add the reshaping in the middle
            self.VU_1 = nn.Sequential(
                nn.Linear(128, int(256 * (self.enc_dim[0] // 2) * (self.enc_dim[1] // 2) * (self.enc_dim[2] // 2))),
                nn.ReLU()
            )
            self.VU_2 = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            self.VUp2 = nn.Sequential(
                nn.Conv3d(256, 128, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock2 = nn.Sequential(
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, padding=1)
            )

            self.VUp1 = nn.Sequential(
                nn.Conv3d(128, 64, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock1 = nn.Sequential(
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, padding=1)
            )

            self.VUp0 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear')
            )

            # Do not forget to add input after this
            self.VBlock0 = nn.Sequential(
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=3, padding=1)
            )

            self.Vend = nn.Conv3d(32, in_channels, kernel_size=1)
        
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
        if self.HR_layers == 0:
            dec_out = self.D3(enc_out_2 + dec_out)
            dec_out = self.D4(enc_out_1 + dec_out)
        elif self.HR_layers == 1:
            dec_out = self.D3(enc_out_2 + dec_out)
            dec_out = self.D4(dec_out)
        elif self.HR_layers == 2:
            dec_out = self.D3(dec_out)
            dec_out = self.D4(dec_out)
        else:
            raise TypeError("Invalid Number of HR Layers")
        
        if self.VAE_enable:
            # VAE layers
            VAE_out = self.VD(enc_out_4)
            distr = VAE_out
            #mu, logvar = torch.chunk(VAE_out, 2, dim=1)  # split into two 128-sized vectors
            VAE_out = self.VDraw(VAE_out)
            VAE_out = self.VU_1(VAE_out)
            VAE_out = VAE_out.view(-1, 256, self.enc_dim[0] // 2, self.enc_dim[1] // 2, self.enc_dim[2] // 2)
            VAE_out = self.VU_2(VAE_out)
            VAE_out = self.VUp2(VAE_out)
            VAE_out = VAE_out + self.VBlock2(VAE_out)
            VAE_out = self.VUp1(VAE_out)
            VAE_out = VAE_out + self.VBlock1(VAE_out)
            VAE_out = self.VUp0(VAE_out)
            VAE_out = VAE_out + self.VBlock0(VAE_out)
            VAE_out = self.Vend(VAE_out)

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
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.res_layers(x)


# Handling undivisible sizes
def make_divisible_crop_or_pad(shape, s):
    """
    Factory function.
    shape: tuple (N, M, K)
    s: scalar (power of 2)

    Returns a function that takes a tensor of shape (..., N, M, K)
    and crops or zero-pads so each dimension becomes divisible by s.
    """
    N, M, K = shape

    # Compute target sizes (smallest multiples of s)
    def target_size(x):
        return int(((x + s - 1) // s) * s)  # ceil(x / s) * s

    N2 = target_size(N)
    M2 = target_size(M)
    K2 = target_size(K)

    def transform(tensor):
        """
        tensor: shape (..., N, M, K)
        returns tensor cropped or padded to (..., N2, M2, K2)
        """

        _, _, h, w, d = tensor.shape[-5:]

        # --- CROP if larger ---
        t = tensor
        if h > N2: t = t[..., :N2, :, :]
        if w > M2: t = t[..., :, :M2, :]
        if d > K2: t = t[..., :, :, :K2]

        # --- PAD if smaller ---
        pad_h = max(0, N2 - t.shape[-3])
        pad_w = max(0, M2 - t.shape[-2])
        pad_d = max(0, K2 - t.shape[-1])

        # PyTorch pad order: (d_before, d_after, w_before, w_after, h_before, h_after)
        padding = (0, pad_d, 0, pad_w, 0, pad_h)
        t = F.pad(t, padding)

        return t

    return transform, (N2, M2, K2)

def make_crop_or_pad(in_shape, out_shape):
    """
    in_shape: tuple of input spatial dims, e.g. (N, M, K)
    out_shape: tuple of desired dims,    e.g. (N2, M2, K2)

    Returns:
      transform: function(tensor) → tensor cropped/padded
    """
    assert len(in_shape) == len(out_shape), "Shape ranks must match"

    # For readability
    in_dims  = list(in_shape)
    out_dims = list(out_shape)
    D = len(in_dims)

    def transform(tensor):
        """
        tensor shape: (..., *in_shape)
        returns:      (..., *out_shape)
        """
        t = tensor

        # ----- CROP first -----
        for i in range(D):
            cur = t.shape[-D + i]
            target = out_dims[i]

            if cur > target:
                # Build the slicing tuple
                slices = [slice(None)] * t.ndim
                slices[-D + i] = slice(0, target)
                t = t[tuple(slices)]

        # ----- PAD second -----
        # PyTorch pad expects pairs: (dim_D_before, dim_D_after, ..., dim_1_before, dim_1_after)
        pads = []
        for i in reversed(range(D)):
            cur = t.shape[-D + i]
            target = out_dims[i]
            pad_amount = max(0, target - cur)
            pads.extend([0, pad_amount])   # before=0, after=pad_amount

        if any(pads):
            t = F.pad(t, pads)

        return t

    return transform



# Local Test
if __name__ == "__main__":
    inChans = 4
    seg_outChans = 3
    input_shape = (1, 144, 240, 240)
    VAE_enable = True

    print("Local Test - Reference Net - Our Implementation")
    print("\nReference Network - Done by us - MOD 04")
    print("Decreased Downsampling Network")

    print("\nWith VAE")
    
    print("\n0 HR")
    model_VAE_0HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=0)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_VAE_0HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n1 HR")
    model_VAE_1HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=1)
    
    x = torch.randn(1, 4, 144 // 2, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_VAE_1HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n2 HR")
    model_VAE_2HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=2)
    
    x = torch.randn(1, 4, 144 // 4, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_VAE_2HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\nNo VAE")
    print("\n0 HR")
    model_noVAE_0HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=0)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_noVAE_0HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n1 HR")
    model_noVAE_1HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=1)
    
    x = torch.randn(1, 4, 144 // 2, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_noVAE_1HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n2 HR")
    model_noVAE_2HR = VAE_UNET_3D_M04(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=2)
    
    x = torch.randn(1, 4, 144 // 4, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_noVAE_2HR(x)

    print("Output 1 shape:", out1.shape)

    print("\nReference Network - Done by us - MOD 01")
    print("Asymmetric Network")

    print("\nWith VAE")
    
    print("\n0 HR")
    model_VAE_0HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=0)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_VAE_0HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n1 HR")
    model_VAE_1HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=1)
    
    x = torch.randn(1, 4, 144 // 2, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_VAE_1HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n2 HR")
    model_VAE_2HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=2)
    
    x = torch.randn(1, 4, 144 // 4, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_VAE_2HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\nNo VAE")
    print("\n0 HR")
    model_noVAE_0HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=0)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_noVAE_0HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n1 HR")
    model_noVAE_1HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=1)
    
    x = torch.randn(1, 4, 144 // 2, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_noVAE_1HR(x)

    print("Output 1 shape:", out1.shape)

    print("\n2 HR")
    model_noVAE_2HR = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False, HR_layers=2)
    
    x = torch.randn(1, 4, 144 // 4, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_noVAE_2HR(x)

    print("Output 1 shape:", out1.shape)

    print("\nReference Network - Done by us")

    print("\nWith VAE")
    model_VAE = REF_VAE_UNET_3D(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=True)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_VAE(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\nNo VAE")
    model_noVAE = REF_VAE_UNET_3D(in_channels=inChans, input_dim=np.asarray([144, 240, 240], dtype=np.int64), num_classes=4, VAE_enable=False)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_noVAE(x)

    print("Output 1 shape:", out1.shape)
    