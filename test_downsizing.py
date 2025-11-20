import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from src.downsample import downsample, filter_downsample
from src.img_wavelet import img_wavelet
from src.degradate_reconstruct import transform_downsample_reconstruct

import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from src.data_preparation import BRATS_dataset

# ==== CONFIG ====
# Folder name under training_results/ that contains params.pkl
label = "MOD01_Test01"   # <-- change this to your run folder name

data_path = "../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

# ==== LOAD PARAMS ====
results_path = Path("training_results") / label
with open(results_path / "params.pkl", "rb") as f:
    params = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== BUILD DATASET & LOADER ====
dataset = BRATS_dataset(data_path, device, params)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get a single batch
out_imgs, inp_imgs, mask = next(iter(loader))

# Assume out_imgs shape is [B, D, H, W]
central_index = out_imgs.shape[1] // 2
central_slice = out_imgs[:, central_index, :, :]  # shape [1, H, W]

print("central_slice shape:", central_slice.shape)

# If you need a NumPy array for downstream task:
central_slice_ = central_slice.squeeze(0).cpu().numpy()
    


phantom = shepp_logan_phantom()
phantom = phantom.astype(np.float32)
phantom = resize(phantom, (240, 240), anti_aliasing=True).astype(np.float32)
phantom = central_slice

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

