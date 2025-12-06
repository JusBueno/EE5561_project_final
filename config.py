import sys
import numpy as np
import argparse


class Configs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training Parameters")

        # Required positional argument
        self.parser.add_argument("folder", type=str,
                                 help="Folder path to save training session")

        # Simple hyperparameters
        self.parser.add_argument("--net", type=str, default="REF_US")
        self.parser.add_argument("--num_epochs", type=int, default=300)
        self.parser.add_argument("--LR", type=float, default=1e-4)
        self.parser.add_argument("--batch_size", type=int, default=1)

        # Loss weight parameters
        self.parser.add_argument("--k1", type=float, default=0.1)
        self.parser.add_argument("--k2", type=float, default=0.1)

        # Downsampling
        self.parser.add_argument("--downsamp_type", type=str, default="bilinear")
        self.parser.add_argument("--ds_ratio", type=int, default=1)

        # Dataset parameters
        self.parser.add_argument("--slab_dim", type=int, default=144)
        #self.parser.add_argument("--slabs_per_volume", type=int, default=1)
        #self.parser.add_argument("--num_volumes", type=int, default=369)
        self.parser.add_argument("--fusion", type=str, default="None")

        # Boolean toggles
        self.parser.add_argument("--crop", action="store_true")
        self.parser.add_argument("--no_crop", action="store_false", dest="crop")
        self.parser.set_defaults(crop=True)

        self.parser.add_argument("--VAE_enable", action="store_true")
        self.parser.add_argument("--VAE_disable", action="store_false", dest="VAE_enable")
        self.parser.set_defaults(VAE_enable=True)

        self.parser.add_argument("--UNET_enable", action="store_true")
        self.parser.add_argument("--UNET_disable", action="store_false", dest="UNET_enable")
        self.parser.set_defaults(UNET_enable=True)

        self.parser.add_argument("--VAE_warmup", action="store_true")
        self.parser.set_defaults(VAE_warmup=False)

        resume_group = self.parser.add_mutually_exclusive_group()
        resume_group.add_argument("--resume", action="store_true",
                                  help="Resume training (default)")
        resume_group.add_argument("--start_new", action="store_false", dest="resume",
                                  help="Start a new training run")
        self.parser.set_defaults(resume=True)


    def parse(self):
        cfg = self.parser.parse_args()

        possible_nets = ["REF", "REF_US", "VAE_3D", "VAE_2D"]

        # Additional derived config values
        cfg.train_ratio = 0.8
        cfg.validation = True
        cfg.save_model_each_epoch = True
        cfg.HR_layers = int(np.log2(cfg.ds_ratio)) if cfg.ds_ratio > 0 else 0

        cfg.threeD = cfg.net in ["REF", "REF_US", "VAE_3D"]
        cfg.logvar_out = cfg.net in ["VAE_2D", "REF_US", "VAE_3D"]
        cfg.data_shape = [155, 240, 240]
        cfg.crop_size = [cfg.slab_dim, 160, 224]
        cfg.modality_index = 0
        cfg.augment = True
        cfg.binary_mask = False
        cfg.slabs_per_volume = 1 if cfg.threeD else 10
        cfg.num_volumes = 369
        cfg.val_patience = 10
        cfg.start_epoch = 0 #in case of resume

        if cfg.net not in possible_nets:
            sys.exit(f"Error: network '{cfg.net}' is not implemented.")

        return cfg
    
def save_configs(cfg, path):
        """
        Save all configuration parameters to a text file.
        Each line will be: <parameter>: <value>
        """
        with open(path, "w") as f:
            for key, value in vars(cfg).items():
                f.write(f"{key}: {value}\n")

#Choose parameters
class Training_Parameters:
    def __init__(self, net = "REF", VAE_enable = True, UNET_enable = True, num_epochs = 300, LR = 1e-4, batch = 1, degradation_type = 'downsampling',
                 downsamp_type = 'bilinear', ds_ratio = 1, crop = True, VAE_warmup = False, fusion = "None"):
        
        #Choose a network
        possible_nets = ["REF", "REF_US", "VAE_3D", "VAE_2D"]
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
        self.train_ratio = 0.8           #What ratio of dataset for training (Training ratio = 1 - validation ratio)
        self.validation = True              #Whether you want validation each epoch
        self.save_model_each_epoch = True   #Save model and training parameters every epoch
        self.crop = crop
        
        #Choose which type of downsampling/degradation
        self.degradation_type = degradation_type
        self.downsamp_type = downsamp_type  #Type of downsampling
        self.ds_ratio = ds_ratio              #Downsampling factor (if doing downsamplin at all)
        self.HR_layers = np.log2(self.ds_ratio)
        
        
        #Data preparation parameters (choose dataset size as slabs_per_volume * num_volumes)
        self.threeD = self.net in ["REF", "REF_US", "VAE_3D"] #Use volume dimension
        self.num_volumes = 2  #Maximum = 369 for the training dataset
        self.data_shape = [155,240,240] #Original data shape [Height x Width x Depth]
        self.crop_size = [self.slab_dim,240,240] #Used data shape [Depth x Height x Width]
        self.modality_index = 0 #If single modality, which one to choose
        self.augment = True     #Perform data augmentation or not
        self.binary_mask = False 
        
        
        if self.net not in possible_nets:
            sys.exit(f"Error: network {self.net} is not implemented")
            sys.exit(1) # Exit with an error code
                          
                          
          
 
        