import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from scipy import ndimage
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    Dataset class for the brain tumor segmentation data using a 2.5D slice setup
    
    INPUTS:
    dataset_path: path leading to the "MICCAI_BraTS2020_TrainingData" directory
    device: 
    params: Training_Parameters instance described above
    """
    
    def __init__(self, dataset_path, device, params):
        
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.slab_dim = params.slab_dim
        self.ds_ratio = params.ds_ratio
        self.data_shape = params.data_shape
        self.downsamp_type = params.downsamp_type
        self.output_dim = [self.slab_dim,240,240]
        self.input_dim = [self.slab_dim//self.ds_ratio,240//self.ds_ratio,240//self.ds_ratio]
        self.augment = params.augment
        self.binary_mask = params.binary_mask
        self.volume_dim = params.volume_dim
        self.modality_index = params.modality_index
        
        
        if params.slabs_per_volume >= self.data_shape[2] - 2*(self.slab_dim//2):
            self.slabs_per_volume = self.data_shape[2] - 2*(self.slab_dim//2)
        else:
            self.slabs_per_volume = params.slabs_per_volume
        
        #If using a few slices per volume, separate the indices out
        self.slice_indices = [ ( (i+1) * self.data_shape[2]  ) // (self.slabs_per_volume+1) for i in range(self.slabs_per_volume) ]
        
        
        subdir_list = [p for p in self.dataset_path.iterdir() if p.is_dir()]
        
        if(len(subdir_list) > params.num_volumes):
            subdir_list = subdir_list[:params.num_volumes]
        
        self.subdir_list = subdir_list
        self.num_volumes = len(subdir_list)
        
        #Total length of the dataset = number of 2.5D slices * number of volumes
        self.length = self.slabs_per_volume * self.num_volumes
         
        
    def __len__(self):
        return self.length
    
    def zscore(self, data):
        non_zero_locs = data>0
        non_zero_data = data[non_zero_locs]
        
        if non_zero_data.size == 0:
            return data
        
        mean = np.mean(non_zero_data)
        std = np.std(non_zero_data)
        if std > 0:
            zscored_data =  (non_zero_data - mean) / std
        else:
            zscored_data = 0
            
        data[non_zero_locs] = zscored_data
        
        return data
        
    
    def downsize(self, img):
        #Downsize in each channel
        ds = 1/float(self.ds_ratio)
        if self.downsamp_type == 'bicubic':
            downscaled_image = ndimage.zoom(img, (ds, ds, ds), order=3)
        if self.downsamp_type == 'bilinear':
            downscaled_image = ndimage.zoom(img, (ds, ds, ds), order=1)

        return downscaled_image

    def __getitem__(self, idx):
        #Each idx will get one 2.5D slice of a particular volume
        volume_idx = idx // self.slabs_per_volume
        slice_idx_temp = idx % self.slabs_per_volume
        slice_idx = self.slice_indices[slice_idx_temp]
        
        if self.slab_dim%2 == 0:
            slice_range = np.arange(slice_idx - self.slab_dim//2, slice_idx + self.slab_dim//2)
        else:
            slice_range = np.arange(slice_idx - self.slab_dim//2, slice_idx + self.slab_dim//2+1)
        
        
        volume_path = self.subdir_list[volume_idx]
        file_list = [p for p in volume_path.iterdir() if p.is_file()]
        
        #List of all volumes
        vol_list = [nib.load(file_list[i]).get_fdata() for i in range(5)]
        
        #Get only the needed 2.5D slice
        data_list = [vol[:,:,slice_range] for vol in vol_list]
        data_list = [img.transpose(2,0,1) for img in data_list] #Channel dim first
        
        #Get segmentation mask from the list
        mask_3d = data_list.pop(1)
 
        #Normalize each 2.5D slice
        img_list = [self.zscore(img) if img.max()>0 else img for img in data_list]
    
        
        #Apply augmentations
        if(self.augment):
            intensity_shift = np.random.uniform(-0.1, 0.1, size=(4,))
            intensity_scale = np.random.uniform(0.9, 1.1, size=(4,))
            axis_flip = np.random.binomial(n=1, p=0.5, size=(3,))

            for i, img in enumerate(img_list):
                img_list[i] += intensity_shift[i]
                img_list[i] *= intensity_scale[i]
                for j, ax in enumerate(axis_flip):
                    if ax: img_list[i] = np.flip(img_list[i], axis = j).copy()
                        
            for j, ax in enumerate(axis_flip):
                if ax: mask_3d = np.flip(mask_3d, axis = j).copy()
                                       
        #Downsample each image (if needed)
        inp_img_list = [self.downsize(img) for img in img_list]
        
        class_list = [1,2,4] #Specific to our dataset, classes skip 3 for some reason
        
        #If binary mask, we only have one channel
        if self.binary_mask:
            mask[mask>=2] = 1
        else:
        #Otherwise, create a multi channel mask with 1s in each class, each channel
            full_mask = np.zeros((3,self.slab_dim,240,240), dtype = int)
            for i in range(0,3):
                temp_mask = np.zeros_like(mask_3d)
                temp_mask[mask_3d == class_list[i]] = 1
                
                full_mask[i,:,:,:] = temp_mask
            mask_3d = full_mask
                
        out_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32) for img in img_list]
        inp_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32)  for img in inp_img_list]
        
        
        mask_3d = torch.from_numpy(mask_3d).to(self.device).to(torch.float32)
        mask_2d = mask_3d[:,self.slab_dim//2,:,:].squeeze()
         
        
        if self.volume_dim:
            vol_out_img = torch.zeros((4, self.output_dim[0], self.output_dim[1], self.output_dim[2]), device = device)
            vol_inp_img = torch.zeros((4, self.input_dim[0], self.input_dim[1], self.input_dim[2]), device = device)
            
            for i in range(4):

                vol_out_img[i, :, :, :] = out_img_list[i]
                vol_inp_img[i, :, :, :] = inp_img_list[i]
                
            return vol_out_img, vol_inp_img, mask_3d
        
        if(self.modality_index is not None):
            return out_img_list[self.modality_index], inp_img_list[self.modality_index], mask_2d
        
        #Image list contains 2.5D slices of: flair, t1, t1ce, t2 (in order)
        return out_img_list_2d, inp_img_list_2d, mask_2d