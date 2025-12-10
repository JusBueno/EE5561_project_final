import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def downsample_2d(img, factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_t = torch.as_tensor(img, dtype=torch.float32).to(device)
    return img_t[::factor, ::factor]

def filter_downsample_2d(img, factor):
    # Initial setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_t = torch.as_tensor(img, dtype=torch.float32).to(device)
    img_4d = img_t.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
    
    sigma = 2 * factor / 6.0 # from scikit-image documentation
    kernel_size = 2 * factor + 1 # to keep kernel size odd
    
    # Reflection padding same convolution 
    img_t_filtered = gaussian_blur(img_4d, kernel_size=kernel_size, sigma=sigma)
    img_filtered = img_t_filtered.squeeze(0).squeeze(0) # Back to original dimension
    return img_filtered[::factor, ::factor]

def downsample_3d(img_3d, factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_3d_t = torch.as_tensor(img_3d, dtype=torch.float32).to(device)
    return img_3d_t[::factor, ::factor, ::factor]

def filter_downsample_3d(img_3d, factor):
    # Initial setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_t = torch.as_tensor(img_3d, dtype=torch.float32, device=device)
    img_5d = img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, d, h, w)

    sigma = 2 * factor / 6.0 # from scikit-image documentation
    kernel_size = 2 * factor + 1 # to keep kernel size odd
    
    # 1D Centered Gaussian
    x = torch.arange(kernel_size, device=device, dtype=img_t.dtype) - kernel_size // 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)

    # 3D Gaussian (outer product)
    gaussian_3d = (gaussian_1d[:, None, None] * gaussian_1d[None, :, None] * gaussian_1d[None, None, :])
    gaussian_3d = gaussian_3d/ gaussian_3d.sum() # Normalize
    gaussian_3d_5d = gaussian_3d.unsqueeze(0).unsqueeze(0) # (1, 1, d, h, w)

    # Filter
    img_5d_filtered = F.conv3d(img_5d, gaussian_3d_5d, padding="same")
    img_3d_filtered = img_5d_filtered.squeeze(0).squeeze(0)
    return img_3d_filtered[::factor, ::factor, ::factor] # Downsample
    