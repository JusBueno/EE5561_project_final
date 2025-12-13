import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import ndimage
from src.random_crop_fun import crop_3d
from src.img_wavelet import img_wavelet, img_wavelet_3d
from src.downsample import *
from config import Configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    BRATS 2020 3D or 2.5D dataset
    """
    def __init__(self, dataset_path, device, params, fixed_crop = True):
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.params = params
        self.fixed_crop = fixed_crop
        
        # Output / input dimensions
        if not self.params.crop:
            self.output_dim = [3, self.params.slab_dim, 240, 240]
            self.input_dim  = [4, self.params.slab_dim // self.params.ds_ratio,
                            240 // self.params.ds_ratio,
                            240 // self.params.ds_ratio]
        else:
            self.output_dim = [3] + self.params.crop_size
            self.input_dim  = [4] + [n//self.params.ds_ratio for n in self.params.crop_size]


        # Compute slabs per volume
        max_slabs = self.params.data_shape[0] - 2*(self.params.slab_dim//2)
        self.slabs_per_volume = min(self.params.slabs_per_volume, max_slabs)

        # Slice index list
        self.slice_indices = [
            ((i+1) * self.params.data_shape[0]) // (self.params.slabs_per_volume + 1)
            for i in range(self.params.slabs_per_volume)
        ]

        # Subdirectories (one per volume)
        subs = [p for p in self.dataset_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
        if len(subs) > self.params.num_volumes:
            subs = subs[:self.params.num_volumes]

        self.subdirs = subs
        self.num_volumes = len(subs)
        self.length = self.slabs_per_volume * self.params.num_volumes


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
        Nc, Nd, Nx, Ny = img.shape
        d = self.params.ds_ratio
        dinv = 1 / float(d)
        s = self.params.downsamp_type

        #If 3D, downsample the whole volume
        if self.params.threeD:
            img_ds = np.zeros([Nc, Nd//d, Nx//d, Ny//d], dtype=img.dtype)
            for i in range(Nc):
                if(s == "bicubic"): img_ds[i] = ndimage.zoom(img[i], (dinv, dinv, dinv), order=3)
                if(s == "bilinear"): img_ds[i] = ndimage.zoom(img[i], (dinv, dinv, dinv), order=1)
                if(s == "wavelet"): img_ds[i] = img_wavelet_3d(img[i], d).cpu()
                if(s == "ds"): img_ds[i] = downsample_3d(img[i], d).cpu()
                if(s == "ds_filter"): img_ds[i] = filter_downsample_3d(img[i], d).cpu()
        #If 2D, downsample each slice individually
        else:
            img_ds = np.zeros([Nc, Nd, Nx//d, Ny//d], dtype=img.dtype)
            for i in range(Nc):
                for j in range(Nd):
                    if(s == "bicubic"): img_ds[i,j] = ndimage.zoom(img[i,j], (dinv, dinv), order=3)
                    if(s == "bilinear"): img_ds[i,j] = ndimage.zoom(img[i,j], (dinv, dinv), order=1)
                    if(s == "wavelet"): img_ds[i,j] = img_wavelet(img[i,j], d).cpu()
                    if(s == "ds"): img_ds[i,j] = downsample_2d(img[i,j], d).cpu()
                    if(s == "ds_filter"): img_ds[i,j] = filter_downsample_2d(img[i,j], d).cpu()
        return img_ds
    

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
        v_idx = idx // self.params.slabs_per_volume
        s_idx = self.slice_indices[idx % self.params.slabs_per_volume]

        # Determine slice range
        half = self.params.slab_dim // 2
        if self.params.slab_dim % 2 == 0:
            sl = np.arange(s_idx - half, s_idx + half)
        else:
            sl = np.arange(s_idx - half, s_idx + half + 1)


        # Load volumes (T1, mask, etc.)
        files = list(self.subdirs[v_idx].glob("*.nii*"))

        # Explicit mask detect
        mask_file = [f for f in files if "seg" in f.name.lower()][0]
        mask = nib.load(mask_file).get_fdata()
        mask = mask[:,:,sl].transpose(2, 0, 1)

        # All non-mask modalities
        vol_files = [f for f in files if "seg" not in f.name.lower()]
        vols = [nib.load(f).get_fdata() for f in sorted(vol_files)]

        # Extract slabs and reorder to [D,H,W]
        slabs = [vol[:, :, sl].transpose(2, 0, 1) for vol in vols]

        # Normalize images
        imgs = [self.zscore(x) if x.max() > 0 else x for x in slabs]

        # Augmentation
        if self.params.augment:
            imgs, mask = self.augment_pair(imgs, mask)

        # Build 3-channel mask (BRATS uses labels 1,2,4)
        classes = [1, 2, 4]
        mask = np.stack([(mask == c).astype(np.float32) for c in classes], axis=0)
        imgs = np.stack(imgs) 
        
        # Apply cropping
        if self.params.crop:
            imgs, mask = crop_3d(imgs, mask, self.params.crop_size, fixed=self.fixed_crop)

        imgs_ds = self.downsize(imgs)

        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        imgs_ds = torch.tensor(imgs_ds, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        if not self.params.threeD:
            Nc, Nz, Nx, Ny = imgs.shape
            Nc, Nz2, Nx2, Ny2 = imgs_ds.shape
            imgs = imgs.view(Nc*Nz,Nx,Ny)
            imgs_ds = imgs_ds.view(Nc*Nz2,Nx2,Ny2)
            mask = mask[:,mask.shape[1]//2,:,:]
        
        return imgs, imgs_ds, mask