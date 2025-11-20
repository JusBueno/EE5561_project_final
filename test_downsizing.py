import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from src.downsample import downsample, filter_downsample
from src.img_wavelet import img_wavelet
from src.degradate_reconstruct import transform_downsample_reconstruct


phantom = shepp_logan_phantom()
phantom_original = phantom.astype(np.float32)
phantom = resize(phantom_original, (240, 240), anti_aliasing=True).astype(np.float32)

img_down = downsample(phantom, 2)
img_filtered_down = filter_downsample(phantom, 2)
img_wav = img_wavelet(phantom)

# Plots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(phantom, cmap="gray")
axes[0].set_title("Original 240x240")
axes[0].axis("off")

axes[1].imshow(img_down, cmap="gray")
axes[1].set_title("Decimation by 2")
axes[1].axis("off")

axes[2].imshow(img_filtered_down, cmap="gray")
axes[2].set_title("Antialias-filtered and decimation by 2")
axes[2].axis("off")

axes[3].imshow(img_wav, cmap="gray")
axes[3].set_title("First wavelet level")
axes[3].axis("off")

plt.tight_layout()
#plt.show()
plt.savefig('./training_results/downsizing_comparison.png')

