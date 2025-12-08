import torch
import torch.nn.functional as F

# Helpers
def _to_tensor(x, device=None, dtype=torch.float32):
    """Convert numpy array or tensor to torch.Tensor on device."""
    if isinstance(x, torch.Tensor):
        t = x.to(dtype)
    else:
        t = torch.as_tensor(x, dtype=dtype)

    if device is not None:
        t = t.to(device)
    return t


def _default_device(device):
    if device is not None:
        return device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def correlate_2d(img, kernel, device=None):
    """
    Same-output-size 2D correlation using PyTorch conv2d (which is correlation).
    img: (H, W)
    kernel: (Kh, Kw)
    Returns: (H, W) torch.Tensor on device
    """
    device = _default_device(device)

    img_t = _to_tensor(img, device=device)
    ker_t = _to_tensor(kernel, device=device)

    # Batch and channel dims: (N=1, C=1, H, W)
    img_4d = img_t.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    ker_4d = ker_t.unsqueeze(0).unsqueeze(0) # (1, 1, Kh, Kw)

    # Same padding
    pad_h = ker_t.shape[-2] // 2
    pad_w = ker_t.shape[-1] // 2

    out = F.conv2d(img_4d, ker_4d, padding=(pad_h, pad_w)) # (1,1,H,W)
    return out.squeeze(0).squeeze(0) # (H, W)


def downsample_2d(img, factor):
    """
    2D downsampling by integer factor.
    Works with torch tensors directly on GPU.
    """
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    return img[..., ::factor, ::factor]


def gaussian_kernel_2d(kernel_size, sigma, device=None, dtype=torch.float32):
    """
    2D isotropic Gaussian kernel of size (kernel_size, kernel_size),
    normalized to sum to 1.

    kernel_size should be odd.
    """
    device = _default_device(device)

    # Symmetric coordinates
    ax = torch.arange(-(kernel_size // 2),
                       kernel_size // 2 + 1,
                       device=device, dtype=dtype)
    xx = ax.view(-1, 1)
    gaussian_1d = torch.exp(-0.5 * (xx / sigma) ** 2)

    # Separable 2D Gaussian
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    gaussian_2d /= gaussian_2d.sum()
    return gaussian_2d


def filter_downsample_2d(img, factor, device=None):
    """
    Antialias filter with Gaussian, then downsample.
    Uses PyTorch conv2d on GPU.
    """
    device = _default_device(device)

    # From scikit-image doc
    sigma = 2 * factor / 6.0
    kernel_size = 2 * factor + 1  # keep kernel size odd

    kernel = gaussian_kernel_2d(kernel_size, sigma, device=device)
    img_t = _to_tensor(img, device=device)

    # filtering + downsampling in a single conv2d with stride=factor
    img_4d = img_t.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    ker_4d = kernel.unsqueeze(0).unsqueeze(0) # (1,1,Kh,Kw)

    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2

    # stride=factor does the downsampling
    out = F.conv2d(img_4d, ker_4d,
                   padding=(pad_h, pad_w),
                   stride=factor) # (1,1,H',W')
    return out.squeeze(0).squeeze(0).detach().cpu().numpy() # (H', W')

def correlate_3d(img, kernel, device=None):
    """
    Same-output-size 3D correlation using PyTorch conv3d.
    img: (D, H, W)
    kernel: (Kd, Kh, Kw)
    Returns: (D, H, W) torch.Tensor on device
    """
    device = _default_device(device)

    img_t = _to_tensor(img, device=device)
    ker_t = _to_tensor(kernel, device=device)

    # Add batch and channel dims: (N=1, C=1, D, H, W)
    img_5d = img_t.unsqueeze(0).unsqueeze(0)
    ker_5d = ker_t.unsqueeze(0).unsqueeze(0)

    pad_d = ker_t.shape[0] // 2
    pad_h = ker_t.shape[1] // 2
    pad_w = ker_t.shape[2] // 2

    out = F.conv3d(img_5d, ker_5d,
                   padding=(pad_d, pad_h, pad_w)) # (1,1,D,H,W)
    return out.squeeze(0).squeeze(0) # (D, H, W)


def downsample_3d(img, factor):
    """
    3D downsampling by integer factor on each axis.
    """
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    return img[..., ::factor, ::factor, ::factor]


def gaussian_kernel_3d(kernel_size, sigma, device=None, dtype=torch.float32):
    """
    3D isotropic Gaussian kernel (kernel_size^3), normalized to sum 1.
    """
    device = _default_device(device)

    ax = torch.arange(-(kernel_size // 2),
                       kernel_size // 2 + 1,
                       device=device, dtype=dtype) # length = kernel_size
    g1d = torch.exp(-0.5 * (ax / sigma) ** 2)

    # Outer products to get separable 3D Gaussian
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    g3d /= g3d.sum()
    return g3d


def filter_downsample_3d(img, factor, device=None):
    """
    Antialias filter with Gaussian in 3D, then downsample.
    Uses conv3d with stride=factor.
    """
    device = _default_device(device)

    sigma = 2 * factor / 6.0
    kernel_size = 2 * factor + 1

    kernel = gaussian_kernel_3d(kernel_size, sigma, device=device)
    img_t = _to_tensor(img, device=device)

    img_5d = img_t.unsqueeze(0).unsqueeze(0) # (1,1,D,H,W)
    ker_5d = kernel.unsqueeze(0).unsqueeze(0)

    pad_d = kernel.shape[0] // 2
    pad_h = kernel.shape[1] // 2
    pad_w = kernel.shape[2] // 2

    out = F.conv3d(img_5d, ker_5d,
                   padding=(pad_d, pad_h, pad_w),
                   stride=factor)
    return out.squeeze(0).squeeze(0).detach().cpu().numpy()
