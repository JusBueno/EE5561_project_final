import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from src.downsample import filter_downsample
from src.img_wavelet import img_wavelet
from src.degradate_reconstruct import transform_downsample_reconstruct


phantom = shepp_logan_phantom()
phantom = phantom.astype(np.float32)
phantom = resize(phantom, (256, 256), anti_aliasing=True).astype(np.float32)

img_filtered_down = filter_downsample(phantom, 2)
img_wav = img_wavelet(phantom)
img_tv_rec = transform_downsample_reconstruct(phantom, 0.3)
img_tv_rec = np.real(img_tv_rec)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_filtered_down, cmap="gray")
axes[0].set_title("Decimation by 2")
axes[0].axis("off")

axes[1].imshow(img_wav, cmap="gray")
axes[1].set_title("First wavelet level")
axes[1].axis("off")

axes[2].imshow(img_tv_rec, cmap="gray")
axes[2].set_title("Reconstructed from sampled Fourier space")
axes[2].axis("off")

plt.tight_layout()
plt.show()
plt.savefig('./training_results')

