import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import ndimage
from src.random_crop_fun import crop_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    BRATS 2020 2.5D dataset
    """
    def __init__(self, dataset_path, device, params, fixed_crop = False):
        self.dataset_path = Path(dataset_path)
        self.device = device
        
        # Core params
        self.slab_dim = params.slab_dim
        self.ds_ratio = params.ds_ratio
        self.data_shape = params.data_shape
        self.downsamp_type = params.downsamp_type
        self.augment = params.augment
        self.binary_mask = params.binary_mask
        self.volume_dim = params.volume_dim
        self.modality_index = params.modality_index
        self.crop = params.crop
        self.crop_size = params.crop_size
        self.fixed_crop = fixed_crop  
        
        # Output / input dimensions
        if not self.crop:
            self.output_dim = [3, self.slab_dim, 240, 240]
            self.input_dim  = [4, self.slab_dim // self.ds_ratio,
                            240 // self.ds_ratio,
                            240 // self.ds_ratio]
        else:
            self.output_dim = [3] + self.crop_size
            self.input_dim  = [4] + [n//self.ds_ratio for n in self.crop_size]

        # Compute slabs per volume
        max_slabs = self.data_shape[2] - 2*(self.slab_dim//2)
        self.slabs_per_volume = min(params.slabs_per_volume, max_slabs)

        # Slice index list
        self.slice_indices = [
            ((i+1) * self.data_shape[2]) // (self.slabs_per_volume + 1)
            for i in range(self.slabs_per_volume)
        ]

        # Subdirectories (one per volume)
        subs = [p for p in Path(dataset_path).iterdir() if p.is_dir()]
        if len(subs) > params.num_volumes:
            subs = subs[:params.num_volumes]

        self.subdirs = subs
        self.num_volumes = len(subs)
        self.length = self.slabs_per_volume * self.num_volumes


    def __len__(self):
        return self.length


    @staticmethod
    def zscore(data):
        mask = data > 0
        if not np.any(mask):
            return data
        vals = data[mask]
        m, s = vals.mean(), vals.std()
        data[mask] = (vals - m) / s if s > 0 else 0
        return data


    def downsize(self, img):
        ds = 1 / float(self.ds_ratio)
        order = 3 if self.downsamp_type == "bicubic" else 1
        return ndimage.zoom(img, (ds, ds, ds), order=order)


    def augment_pair(self, imgs, mask):
        """
        Apply shared flips + per-image intensity transforms.
        """
        # Global axis flips
        flips = np.random.binomial(1, 0.5, size=3)

        # Per-modality shifts/scales
        shifts = np.random.uniform(-0.1, 0.1, size=len(imgs))
        scales = np.random.uniform(0.9, 1.1, size=len(imgs))

        out = []
        for i, img in enumerate(imgs):
            img = img + shifts[i]
            img = img * scales[i]
            for ax in np.where(flips)[0]:
                img = np.flip(img, axis=ax).copy()
            out.append(img)

        # Mask flips
        for ax in np.where(flips)[0]:
            mask = np.flip(mask, axis=ax).copy()

        return out, mask


    def __getitem__(self, idx):
        # Identify which volume + which slab
        v_idx = idx // self.slabs_per_volume
        s_idx = self.slice_indices[idx % self.slabs_per_volume]

        # Determine slice range
        half = self.slab_dim // 2
        if self.slab_dim % 2 == 0:
            sl = np.arange(s_idx - half, s_idx + half)
        else:
            sl = np.arange(s_idx - half, s_idx + half + 1)

        # Load volumes (T1, mask, etc.)
        files = list(self.subdirs[v_idx].glob("*.nii*"))

        # Explicit mask detect
        mask_file = [f for f in files if "seg" in f.name.lower()][0]
        mask = nib.load(mask_file).get_fdata()
        mask = mask.transpose(2, 0, 1)

        # All non-mask modalities
        vol_files = [f for f in files if "seg" not in f.name.lower()]
        vols = [nib.load(f).get_fdata() for f in sorted(vol_files)]

        # Extract slabs and reorder to [D,H,W]
        slabs = [vol[:, :, sl].transpose(2, 0, 1) for vol in vols]

        # Normalize images
        imgs = [self.zscore(x) if x.max() > 0 else x for x in slabs]

        # Augmentation
        if self.augment:
            imgs, mask = self.augment_pair(imgs, mask)

        # Build 3-channel mask (BRATS uses labels 1,2,4)
        classes = [1, 2, 4]
        mask = np.stack([(mask == c).astype(np.float32) for c in classes], axis=0)
        imgs = np.stack(imgs) 
        
        # Apply cropping
        if self.crop:
            imgs, mask = crop_3d(imgs, mask, self.crop_size, fixed=self.fixed_crop)

        #imgs_ds = self.downsize(imgs)
        imgs_ds = imgs

        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        imgs_ds = torch.tensor(imgs_ds, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        # Full-volume mode
        if self.volume_dim:
            return imgs, imgs_ds, mask

        # # Single-modality mode
        # if self.modality_index is not None:
        #     mid = self.modality_index
        #     mask_mid = mask_t[:, self.slab_dim // 2]  # 2D center slice
        #     return imgs_t[mid], imgs_ds_t[mid], mask_mid

        raise RuntimeError("Dataset configuration is incomplete.")