#Copied from https://github.com/amir-aghdam/3D-Brain-Tumor-Segmentation-using-AutoEncoder-Regularization/tree/main
#Network definition and loss function defitions were pasted here
# Modified for downsampled input
# Skip upsampling Network

import torch
from torch import nn
import torch.nn.functional as F

class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()
        
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans, 
                     out_channels=outChans, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate,inplace=True)
            
    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    '''
    Encoder block
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out


class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
    
    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        if skipx is not None:
            if out.shape != skipx.shape: 
                out = nn.functional.interpolate(out, size=skipx.shape[-3:], mode=self.mode, align_corners=self.align_corners)
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)
        
        return out



class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)            
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out



class OutputTransition(nn.Module):
    '''
    Decoder output layer 
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.sigmoid
        
    def forward(self, x):
        return self.actv1(self.conv1(x))



def VDraw(x):
    x = torch.abs(x)
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.normal.Normal(x[:,:128], x[:,128:]).sample()


class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''
    def __init__(self, inChans=256, outChans=256, dense_features=(10,12,8), stride=2, kernel_size=3, padding=1, activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()
        
        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8,num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*dense_features[0]*dense_features[1]*dense_features[2], out_features=inChans)
        self.dense2 = nn.Linear(in_features=midChans, out_features=midChans*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(midChans,outChans)
        
    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = out.view(-1, self.num_flat_features(out))
        out_vd = self.dense1(out)
        distr = out_vd 
        out = VDraw(out_vd)
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((1, 128, self.dense_features[0],self.dense_features[1],self.dense_features[2]))
        out = self.up0(out)
        
        return out, distr
            
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features



class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''
    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)
    
    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out



class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''
    def __init__(self, inChans=256, outChans=4, dense_features=(10,12,8), activation="relu", normalizaiton="group_normalization", mode="trilinear", HR_layers=0):
        super(VAE, self).__init__()
        self.HR_layers = HR_layers

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans, inChans//2)
        self.vd_block1 = VDecoderBlock(inChans//2, inChans//4)
        self.vd_block0 = VDecoderBlock(inChans//4, inChans//8)
        if self.HR_layers == 0:
            self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)
        elif self.HR_layers == 1:
            self.vd_block_HR0 = VDecoderBlock(inChans//8, inChans//16)
            self.vd_end = nn.Conv3d(inChans//16, outChans, kernel_size=1)
        elif self.HR_layers == 2:
            self.vd_block_HR1 = VDecoderBlock(inChans//8, inChans//16)
            self.vd_block_HR0 = VDecoderBlock(inChans//16, inChans//32)
            self.vd_end = nn.Conv3d(inChans//32, outChans, kernel_size=1)
        else:
            raise TypeError("Invalid Number of HR Layers")
        
    def forward(self, x):
        out, distr = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)

        if self.HR_layers == 2:
            out = self.vd_block_HR1(out)
            out = self.vd_block_HR0(out)
            out = self.vd_end(out)
        elif self.HR_layers == 1:
            out = self.vd_block_HR0(out)
            out = self.vd_end(out)
        elif self.HR_layers == 0:
            out = self.vd_end(out)
        else:
            raise TypeError("Invalid Number of HR Layers")

        return out, distr



class NvNet_MOD03(nn.Module):
    def __init__(self, inChans, input_shape, seg_outChans, activation, normalizaiton, VAE_enable, mode, HR_layers = 0):
        super(NvNet_MOD03, self).__init__()
        
        # Original input shape and forced input shape
        self.old_in_shape = (input_shape[1] // (2 ** HR_layers), input_shape[2] // (2 ** HR_layers), input_shape[3] // (2 ** HR_layers))
        self.shape_trans, self.new_in_shape = make_divisible_crop_or_pad(self.old_in_shape, 16)
        
        # some critical parameters
        self.inChans = inChans
        self.input_shape = (input_shape[0], self.new_in_shape[0], self.new_in_shape[1], self.new_in_shape[2])
        self.shape_inv_trans = make_crop_or_pad(self.input_shape, input_shape)
        self.seg_outChans = seg_outChans
        self.activation = activation
        self.normalizaiton = normalizaiton
        self.mode = mode
        self.VAE_enable = VAE_enable
        self.HR_layers = HR_layers # 0, 1, 2
        
        # HR Upsampler
        self.HR_ups1 = LinearUpSampling(32, 16, mode=self.mode)
        self.HR_ups0 = LinearUpSampling(16, 8, mode=self.mode)

        # Encoder Blocks
        self.in_conv0 = DownSampling(inChans=self.inChans, outChans=32, stride=1,dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        
        # Decoder Blocks
        self.de_up2 =  LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up1 =  LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up0 =  LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        
        if self.HR_layers == 0:
            self.de_end = OutputTransition(32, self.seg_outChans)
        elif self.HR_layers == 1:
            self.de_up_HR0 =  LinearUpSampling(32, 16, mode=self.mode)
            self.de_block_HR0 = DecoderBlock(16, 16, activation=self.activation, normalizaiton=self.normalizaiton)
            self.de_end = OutputTransition(16, self.seg_outChans)
        elif self.HR_layers == 2:
            self.de_up_HR1 =  LinearUpSampling(32, 16, mode=self.mode)
            self.de_block_HR1 = DecoderBlock(16, 16, activation=self.activation, normalizaiton=self.normalizaiton)
            self.de_up_HR0 =  LinearUpSampling(16, 8, mode=self.mode)
            self.de_block_HR0 = DecoderBlock(8, 8, activation=self.activation, normalizaiton=self.normalizaiton)
            self.de_end = OutputTransition(8, self.seg_outChans)
        else:
            raise TypeError("Invalid Number of HR Layers")
        
        
        # Variational Auto-Encoder
        if self.VAE_enable:
            self.dense_features = (self.input_shape[1]//16, self.input_shape[2]//16, self.input_shape[3]//16)
            self.vae = VAE(256, outChans=self.inChans, dense_features=self.dense_features, HR_layers=HR_layers)

    def forward(self, x):
        out_init = self.in_conv0(self.shape_trans(x))
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0))) 
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))
        
        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        if self.HR_layers == 0:
            out_end = self.de_end(out_de0)
        elif self.HR_layers == 1:
            out_HR_ups1 = self.HR_ups1(out_en0)
            out_de_HR0 = self.de_block_HR0(self.de_up_HR0(out_de0, out_HR_ups1))
            out_end = self.de_end(out_de_HR0)
        elif self.HR_layers == 2:
            out_HR_ups1 = self.HR_ups1(out_en0)
            out_HR_ups0 = self.HR_ups0(out_HR_ups1)
            out_de_HR1 = self.de_block_HR1(self.de_up_HR1(out_de0, out_HR_ups1))
            out_de_HR0 = self.de_block_HR0(self.de_up_HR0(out_de_HR1, out_HR_ups0))
            out_end = self.de_end(out_de_HR0)
        else:
            raise TypeError("Invalid Number of HR Layers")
        
        out_end = self.shape_inv_trans(out_end)
        
        if self.VAE_enable:
            out_vae, out_distr = self.vae(out_en3)
            out_vae = self.shape_inv_trans(out_vae)
            out_final = torch.cat((out_end, out_vae), 1)
            return out_end, out_vae, out_distr
        
        return out_end
    
    
    
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
        return ((x + s - 1) // s) * s  # ceil(x / s) * s

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
    print("Local Test - Reference Net - MOD03")
    print("Skip Upsampling Network with downsampled input")

    
    print("\nNO HR Layers")
    model_0HR = NvNet_MOD03(inChans=inChans, input_shape=input_shape, seg_outChans=seg_outChans, activation=activation,normalizaiton=normalization,VAE_enable=VAE_enable,mode='trilinear', HR_layers=0)
    
    x = torch.randn(1, 4, 144, 240, 240) 
    out1, out2, out3 = model_0HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)
    
    print("\n1 HR Layers")
    model_1HR = NvNet_MOD03(inChans=inChans, input_shape=input_shape, seg_outChans=seg_outChans, activation=activation,normalizaiton=normalization,VAE_enable=VAE_enable,mode='trilinear', HR_layers=1)
    
    x = torch.randn(1, 4, 144 // 2, 240 // 2, 240 // 2) 
    out1, out2, out3 = model_1HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)

    print("\n2 HR Layers")
    model_2HR = NvNet_MOD03(inChans=inChans, input_shape=input_shape, seg_outChans=seg_outChans, activation=activation,normalizaiton=normalization,VAE_enable=VAE_enable,mode='trilinear', HR_layers=2)
    
    x = torch.randn(1, 4, 144 // 4, 240 // 4, 240 // 4) 
    out1, out2, out3 = model_2HR(x)

    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
    print("Output 3 shape:", out3.shape)