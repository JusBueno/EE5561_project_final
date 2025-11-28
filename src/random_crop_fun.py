import torch

def crop_3d(img, mask, crop_size, fixed=False):
    """
    Crops paired 3D volumes [D,H,W] either centered (fixed=True) or randomly.
    """
    C, D, H, W = img.shape
    cd, ch, cw = crop_size

    assert cd <= D and ch <= H and cw <= W

    if fixed:
        d0 = (D - cd) // 2
        h0 = (H - ch) // 2
        w0 = (W - cw) // 2
    else:
        d0 = torch.randint(0, D - cd + 1, (1,)).item()
        h0 = torch.randint(0, H - ch + 1, (1,)).item()
        w0 = torch.randint(0, W - cw + 1, (1,)).item()

    return (
        img[:,d0:d0+cd, h0:h0+ch, w0:w0+cw],
        mask[:,d0:d0+cd, h0:h0+ch, w0:w0+cw],
    )