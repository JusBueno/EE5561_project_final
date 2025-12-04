import sys
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameters")

    # --- Required positional argument ---
    parser.add_argument("folder", type=str,
                        help="Folder path to save trainining session")


    parser.set_defaults(resume=True)
    parser.add_argument("--start_new", action="store_false", dest="resume", 
                        help="Disable resuming training")
    parser.add_argument("--resume", action="store_true", 
                        help="Enable resuming training (default: True)")
    
    parser.add_argument("--net", type=str, default="REF",
                        help="Network type (default: REF)")

    parser.add_argument("--VAE_enable", action="store_true")
    parser.add_argument("--VAE_disable", action="store_false", dest="VAE_enable")
    parser.set_defaults(VAE_enable=True)

    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Number of epochs (default: 300)")

    parser.add_argument("--LR", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")

    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (default: 1)")

    parser.add_argument("--degradation_type", type=str, default="downsampling",
                        help="Degradation type (default: downsampling)")

    parser.add_argument("--downsamp_type", type=str, default="bilinear",
                        help="Downsampling type (default: bilinear)")

    parser.add_argument("--ds_ratio", type=int, default=1,
                        help="Downsampling ratio (default: 1)")

    parser.add_argument("--fusion", type=str, default="None",
                        help="Downsampling ratio (default: 1)")
    
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--no_crop", action="store_false", dest="crop")
    parser.set_defaults(crop=False)

    parser.add_argument("--UNET_enable", action="store_true")
    parser.add_argument("--UNET_disable", action="store_false", dest="UNET_enable")
    parser.set_defaults(UNET_enable=True)

    parser.add_argument("--VAE_warmup", action="store_true")
    parser.set_defaults(VAE_warmup=False)

    return parser.parse_args()

#Choose parameters
class Training_Parameters:
    def __init__(self, net = "REF", VAE_enable = True, UNET_enable = True, num_epochs = 300, LR = 1e-4, batch = 1, degradation_type = 'downsampling',
                 downsamp_type = 'bilinear', ds_ratio = 1, crop = True, VAE_warmup = False, fusion = "None"):
        
        #Choose a network
        possible_nets = ["REF", "REF_US", "VAE_M01", "VAE_2D"]
        fusion_types = ["None", "Slab", "Modality", "Hybrid"]

        self.net = net
        self.VAE_enable = VAE_enable
        self.UNET_enable = UNET_enable
        self.VAE_warmup = VAE_warmup
        self.fusion = fusion

        #Basic parameters for training
        self.num_epochs = num_epochs    
        self.learning_rate = LR
        self.batch_size = batch
        self.train_ratio = 0.5           #What ratio of dataset for training (Training ratio = 1 - validation ratio)
        self.validation = True              #Whether you want validation each epoch
        self.save_model_each_epoch = True   #Save model and training parameters every epoch
        self.crop = crop
        
        #Choose which type of downsampling/degradation
        self.degradation_type = degradation_type
        self.downsamp_type = downsamp_type  #Type of downsampling
        self.ds_ratio = ds_ratio              #Downsampling factor (if doing downsamplin at all)
        self.HR_layers = np.log2(self.ds_ratio)
        
        
        #Data preparation parameters (choose dataset size as slabs_per_volume * num_volumes)
        self.threeD = self.net in ["REF", "REF_US", "VAE_M01"] #Use volume dimension
        self.slab_dim = 144
        self.slabs_per_volume = 1
        self.num_volumes = 2  #Maximum = 369 for the training dataset
        self.data_shape = [240,240,155] #Original data shape [Height x Width x Depth]
        self.crop_size = [self.slab_dim,240,240] #Used data shape [Depth x Height x Width]
        self.modality_index = 0 #If single modality, which one to choose
        self.augment = True     #Perform data augmentation or not
        self.binary_mask = False 
        
        
        if self.net not in possible_nets:
            sys.exit(f"Error: network {self.net} is not implemented")
            sys.exit(1) # Exit with an error code
                          
                          
          
 
        