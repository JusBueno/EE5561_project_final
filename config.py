import sys
import numpy as np

#Choose parameters
class Training_Parameters:
    def __init__(self, net = "REF", VAE_enable = True, num_epochs = 300, LR = 1e-4, batch = 1, degradation_type = 'downsampling',
                 downsamp_type = 'bilinear', ds_ratio = 1):
        
        #Choose a network
        possible_nets = ["REF", "MOD_01", "MOD_02", "MOD_03"]
        
        self.net = net
        self.VAE_enable = VAE_enable

        #Basic parameters for training
        self.num_epochs = num_epochs    
        self.learning_rate = LR
        self.batch_size = batch
        self.train_ratio = 0.8              #What ratio of dataset for training (Training ratio = 1 - validation ratio)
        self.validation = True              #Whether you want validation each epoch
        self.save_model_each_epoch = True   #Save model and training parameters every epoch
        
        
        
        #Choose which type of downsampling/degradation
        self.degradation_type = degradation_type
        self.downsamp_type = downsamp_type  #Type of downsampling
        self.ds_ratio = ds_ratio              #Downsampling factor (if doing downsamplin at all)
        self.HR_layers = np.log2(self.ds_ratio)
        
        
        
        #Data preparation parameters (choose dataset size as slabs_per_volume * num_volumes)
        self.volume_dim = True #Use volume dimension
        self.slab_dim = 144
        self.slabs_per_volume = 1
        self.num_volumes = 369  #Maximum = 369 for the training dataset
        self.data_shape = [240,240,155]
        self.modality_index = 0 #If single modality, which one to choose
        self.augment = True     #Perform data augmentation or not
        self.binary_mask = False 
        
        
        if self.net not in possible_nets:
            sys.exit(f"Error: network {self.net} is not implemented")
            sys.exit(1) # Exit with an error code
                          
                          
          
 
        