import torch
import torch.nn.functional as F
import math

def img_wavelet(img, factor):
    '''
    Returns 1st or 2nd level "Blur" 2d wavelet of img.
    The output is downsized by 2 if dimesion is even.
    '''
    if factor not in {2, 4}:
        raise ValueError("Factor must be one of {2, 4}")
    
    # Initial setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_t = torch.as_tensor(img, dtype=torch.float32).to(device)
    img_4d = img_t.unsqueeze(0).unsqueeze(0) # adding extra dimension for downstream operations

    # Scaling function is a lowpass filter
    scaling_function = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32) / math.sqrt(2.0)
    # Outer product to get LL component (separability principle)
    LL_filter = torch.outer(scaling_function, scaling_function).unsqueeze(0).unsqueeze(0)

    # Implement wavelet with convolution (filtering interpretation)
    img_downsized = F.conv2d(img_4d, LL_filter, stride=2)
    img_downsized = img_downsized.squeeze(0).squeeze(0)
    if factor == 4:
        img_downsized = img_wavelet(img_downsized, 2)

    return img_downsized

def img_wavelet_3d(img_3d, factor):
    '''
    Returns 1st or 2nd level "Blur" 3d wavelet of img.
    The output is downsized by 2 if dimesion is even.
    If it is odd, output dimension is downsized by 2 and
    rounded down by one
    '''
    if factor not in {2, 4}:
        raise ValueError("Factor must be one of {2, 4}")
    
    # Initial setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_3d_t = torch.as_tensor(img_3d, dtype=torch.float32).to(device)
    img_5d = img_3d_t.unsqueeze(0).unsqueeze(0)

    # Scaling function is a lowpass filter
    scaling_function = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32) / math.sqrt(2.0)
    # Outer product to get LLL component (separability principle)
    LLL_filter = torch.outer(torch.outer(scaling_function, 
                                         scaling_function).reshape(-1), 
                                         scaling_function).reshape(2, 2, 2).unsqueeze(0).unsqueeze(0) 

    # Implement wavelet with convolution (filtering interpretation)
    img_3d_downsized = F.conv3d(img_5d, LLL_filter, stride=2)
    img_3d_downsized = img_3d_downsized.squeeze(0).squeeze(0)
    if factor == 4:
        img_3d_downsized = img_wavelet_3d(img_3d_downsized, 2)

    return img_3d_downsized
