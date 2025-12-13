import torch
import numpy as np
import matplotlib.pyplot as plt
from src.network_3D import VAE_UNET_3D_M01, REF_VAE_UNET_3D
from src.criterion import soft_dice_coeff


def load_model(net_type, VAE_enable = True):
    crop_dim = np.asarray([144, 160, 224], dtype=np.int64)
    if net_type == "REF":
        model = REF_VAE_UNET_3D(in_channels=4, input_dim=crop_dim, num_classes=4, VAE_enable=VAE_enable)
    elif net_type == "VAE_3D":
        model = VAE_UNET_3D_M01(in_channels=4, input_dim=crop_dim, num_classes=4, VAE_enable=VAE_enable, HR_layers = 1)

    return model

#Choose parameters
class DataParams:
    def __init__(self, ds_ratio = 1, downsamp_type = "bilinear"):
        
        self.slab_dim = 144
        self.ds_ratio = ds_ratio
        self.slabs_per_volume = 1
        self.data_shape = [155, 240, 240]
        self.num_volumes = 369
        self.downsamp_type = downsamp_type
        self.threeD = True
        self.augment = True
        self.crop_size = [144, 160, 224]
        self.crop = True


def plot_examples_demo(model, test_dataset, slices, VAE_enable = True):
    
    model.eval()
    print("Plotting results" + "-"*60)
    
    MSE_loss = torch.nn.MSELoss()
    fs = 16 #fontsize
    
    with torch.no_grad():
        j = 0
        for i in slices:
            
            out_img, inp_img, mask = test_dataset[i]
            
            mask = mask.unsqueeze(0)
            out_img = out_img.unsqueeze(0)
            inp_img = inp_img.unsqueeze(0)

            ds = out_img.shape[-1] != inp_img.shape[-1]
            
            seg_out, rec_pred, _ = model(inp_img)


            HR_image = out_img[0,0,out_img.shape[2]//2,:,:].squeeze()
            degrad_image = inp_img[0,0,inp_img.shape[2]//2,:,:].squeeze()
            
            if not VAE_enable:
                seg_out = torch.round(seg_out)
                m = seg_out.shape[2]//2
                seg_out_2d = seg_out[0:1,:,m:m+1,:,:]
                mask_2d = mask[0:1,:,m:m+1,:,:]

                dice_coeff = soft_dice_coeff(seg_out_2d, mask_2d).squeeze()
                
                seg_out_2d = seg_out_2d.squeeze()
                mask_2d = mask_2d.squeeze()
                
                if ds:
                    fig, ax = plt.subplots(1,4,figsize = (13.5,3.5))
                    plt.gray()
                    ax[0].imshow(HR_image.cpu())
                    ax[0].set_title("High-res central slice (FLAIR)", fontsize = fs)
                    ax[1].imshow(degrad_image.cpu())
                    ax[1].set_title("Downsampled central slice", fontsize = fs)
                    ax[2].imshow(mask_2d.cpu().permute(1,2,0))
                    ax[2].set_title("True mask", fontsize = fs)
                    ax[3].imshow(seg_out_2d.cpu().permute(1,2,0))
                    ax[3].set_title(f"Predicted high-res mask \n Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                else:
                    fig, ax = plt.subplots(1,4,figsize = (10,3.5))
                    plt.gray()
                    ax[0].imshow(HR_image.cpu())
                    ax[0].set_title("High-res central slice (FLAIR)", fontsize = fs)
                    ax[1].imshow(mask_2d.cpu().permute(1,2,0))
                    ax[1].set_title("High-res true mask", fontsize = fs)
                    ax[2].imshow(seg_out_2d.cpu().permute(1,2,0))
                    ax[2].set_title(f"Predicted mask \n Dice coeff = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                
                

            else:
                seg_out = torch.round(seg_out)
                m = seg_out.shape[2]//2
                seg_out_2d = seg_out[0:1,:,m:m+1,:,:]
                mask_2d = mask[0:1,:,m:m+1,:,:]


                dice_coeff = soft_dice_coeff(seg_out_2d, mask_2d).squeeze()

                seg_out_2d = seg_out_2d.squeeze()
                mask_2d = mask_2d.squeeze()

                mse_loss = MSE_loss(rec_pred, out_img)
                vae_out_2d = rec_pred[0,0,rec_pred.shape[2]//2,:,:].squeeze()
                if ds:
                    fig, ax = plt.subplots(1,5,figsize = (17,3.5))
                    plt.gray()
                    ax[0].imshow(HR_image.cpu())
                    ax[0].set_title("High-res central slice (FLAIR)", fontsize = fs)
                    ax[1].imshow(degrad_image.cpu())
                    ax[1].set_title("Downsampled central slice", fontsize = fs)
                    ax[2].imshow(mask_2d.cpu().permute(1,2,0))
                    ax[2].set_title("True mask", fontsize = fs)
                    ax[3].imshow(seg_out_2d.cpu().permute(1,2,0))
                    ax[3].set_title(f"Predicted mask \n Dice coeff = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                    ax[4].imshow(vae_out_2d.cpu())
                    ax[4].set_title(f"VAE output | MSE = {mse_loss:.3f}", fontsize = fs)
                else:
                    fig, ax = plt.subplots(1,4,figsize = (13,3.5))
                    plt.gray()
                    ax[0].imshow(HR_image.cpu())
                    ax[0].set_title("High-res central slice (FLAIR)", fontsize = fs)
                    ax[1].imshow(mask_2d.cpu().permute(1,2,0))
                    ax[1].set_title("True mask", fontsize = fs)
                    ax[2].imshow(seg_out_2d.cpu().permute(1,2,0))
                    ax[2].set_title(f"Predicted mask \n Dice coeff = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                    ax[3].imshow(vae_out_2d.cpu())
                    ax[3].set_title(f"VAE output | MSE = {mse_loss:.3f}", fontsize = fs)
                
                
                
                

            for ax in fig.get_axes():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='x', length=0)
                ax.tick_params(axis='y', length=0)

            fig.tight_layout()
            plt.show()