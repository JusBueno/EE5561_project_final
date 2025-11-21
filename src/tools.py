import torch.nn.functional as F

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
