import sys
import numpy as np

#Choose parameters
class Training_Parameters:
    def __init__(self):
        
        #Choose a network
        possible_nets = ["REF", "MOD_01", "MOD_02", "MOD_03"]
        net_3D = ["REF", "MOD_01", "MOD_02", "MOD_03"]

        self.net = "MOD_01"
        self.VAE_enable = True

        
        
  
        #Basic parameters for training
        self.num_epochs = 300      
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.train_ratio = 0.8              #What ratio of dataset for training (Training ratio = 1 - validation ratio)
        self.validation = True              #Whether you want validation each epoch
        self.save_model_each_epoch = True   #Save model and training parameters every epoch
        
        
        
        #Choose which type of downsampling/degradation
        self.degradation_type = 'downsampling'
        self.downsamp_type = 'bilinear'  #Type of downsampling
        self.ds_ratio = 1                #Downsampling factor (if doing downsamplin at all)
        if self.net in net_3D:
            self.downsamp_mode = "3D"
        else:
            self.downsamp_mode = "2D"

        # Number of high resolution layers (0: no downsampling, 1: x2 downsampling, 2: x4 downsampling)
        self.HR_layers = 0
        
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
                          
                          
          
 
        