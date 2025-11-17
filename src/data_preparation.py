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

#Choose parameters
class Training_Parameters:
    def __init__(self):
        
        #Choose network type VAE_2D, UNET_2D, ref_3D, ref_3D_MOD01, ref_3D_MOD02, or ref_3D_MOD03
        #If using a 2D version, volume_dim should be False
        #If using a 3D version, volume_dim should be True. 
        #For 3D version, use num_slices >= 16 and a power of 2
        self.net = "ref_3D"
        
        #Data prep parameters
        self.volume_dim = True           #To use multiple modalities AND 2.5D slabs. This adds volume dim and requires 3dConv()
        self.num_slices = 144             #Number of slices per 2.5D slab
        self.slices_per_volume = 1       #How many slices to use per volume (used to restrict how much data we use)
        self.data_shape = [240,240,155]  #Shape of each volume
        self.downsamp_type = 'bilinear'  #Type of downsampling (maybe we can generalize to any type of degradation)
        self.ds_ratio = 1                #Downsampling factor (if doing downsamplin at all)
        self.num_volumes = 1        #How many volumes from the original dataset to use
        self.cat_modalities = False      #If we want to concatenate MRI modalities along channel dimension           
        self.augment = True             #Perform data augmentation (random scale and shift) or not
        self.binary_mask = False         #Have yes/no singel channel mask instead of 3 channels for tumor types
        self.modality_index = 0          #If using one modality, choose which one
        
        self.validation = True              #Whether you want validation each epoch
        self.save_model_each_epoch = True   #Save model and training parameters every epoch
        self.train_ratio = 0.75              #What ratio of dataset for training (Training ratio = 1 - validation ratio)
        
         
        #Basic parameters
        self.num_epochs = 1               
        self.learning_rate = 1e-4
        self.batch_size = 1

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
        self.num_slices = params.num_slices
        self.ds_ratio = params.ds_ratio
        self.data_shape = params.data_shape
        self.downsamp_type = params.downsamp_type
        self.output_dim = np.array([240,240,self.num_slices], dtype = np.int64)
        self.input_dim = np.array([240//self.ds_ratio,240//self.ds_ratio,self.num_slices], dtype = np.int64)
        self.augment = params.augment
        self.binary_mask = params.binary_mask
        self.cat_modalities = params.cat_modalities
        self.volume_dim = params.volume_dim
        self.modality_index = params.modality_index
        
        
        if params.slices_per_volume >= self.data_shape[2] - 2*(self.num_slices//2):
            self.slices_per_volume = self.data_shape[2] - 2*(self.num_slices//2)
        else:
            self.slices_per_volume = params.slices_per_volume
        
        #If using a few slices per volume, separate the indices out
        self.slice_indices = [ ( (i+1) * self.data_shape[2]  ) // (self.slices_per_volume+1) for i in range(self.slices_per_volume) ]
        
        
        subdir_list = [p for p in self.dataset_path.iterdir() if p.is_dir()]
        
        if(len(subdir_list) > params.num_volumes):
            subdir_list = subdir_list[:params.num_volumes]
        
        self.subdir_list = subdir_list
        self.num_volumes = len(subdir_list)
        
        #Total length of the dataset = number of 2.5D slices * number of volumes
        self.length = self.slices_per_volume * self.num_volumes
         
        
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
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=3)
        if self.downsamp_type == 'bilinear':
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=1)

        return downscaled_image

    def __getitem__(self, idx):
        #Each idx will get one 2.5D slice of a particular volume
        volume_idx = idx // self.slices_per_volume
        slice_idx_temp = idx % self.slices_per_volume
        slice_idx = self.slice_indices[slice_idx_temp] 
   
            
        slice_range = np.arange(slice_idx - self.num_slices//2, slice_idx + self.num_slices//2)
        
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
        class_list = [1,2,4]
        
        #If binary mask, we only have one channel
        if self.binary_mask:
            mask[mask>=2] = 1
        else:
        #Otherwise, create a multi channel mask with 1s in each class, each channel
            full_mask = np.zeros((3,self.num_slices,240,240), dtype = int)
            for i in range(0,3):
                temp_mask = np.zeros_like(mask_3d)
                temp_mask[mask_3d == class_list[i]] = 1
                full_mask[i,:,:,:] = temp_mask
            mask_3d = full_mask
                
        out_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32) for img in img_list]
        inp_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32)  for img in inp_img_list]
        
        
        mask_3d = torch.from_numpy(mask_3d).to(self.device).to(torch.float32)
        mask_2d = mask_3d[:,self.num_slices//2,:,:].squeeze()
        
                

        if self.cat_modalities:
            concat_out_img = torch.zeros((self.num_slices * 4, self.output_dim[0], self.output_dim[1]))
            concat_inp_img = torch.zeros((self.num_slices * 4, self.input_dim[0], self.input_dim[1]))
            for i in range(4):
                concat_out_img[i*self.num_slices : (i+1)*self.num_slices, :, :] = out_img_list[i]
                concat_inp_img[i*self.num_slices : (i+1)*self.num_slices, :, :] = inp_img_list[i]
                
            return concat_img, concat_inp_img, mask
        
        if self.volume_dim:

            vol_out_img = torch.zeros((4, self.num_slices, self.output_dim[0], self.output_dim[1]), device = device)
            vol_inp_img = torch.zeros((4, self.num_slices, self.input_dim[0], self.input_dim[1]), device = device)
            vol_mask = torch.zeros((4, self.num_slices, self.input_dim[0], self.input_dim[1]), device = device)
            for i in range(4):
                vol_out_img[i, :, :, :] = out_img_list[i]
                vol_inp_img[i, :, :, :] = inp_img_list[i]
                
            return vol_out_img, vol_inp_img, mask_3d
        
        if(self.modality_index is not None):
            return out_img_list[self.modality_index], inp_img_list[self.modality_index], mask_2d
        
        #Image list contains 2.5D slices of: flair, t1, t1ce, t2 (in order)
        return out_img_list_2d, inp_img_list_2d, mask_2d