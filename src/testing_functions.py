from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.criterion import *


def test_model(model, test_loader, net_type, VAE_enable = True):
    """
    Testing function for the model
    model: model to be tested
    test_loader: data loader for the test dataset
    VAE: boolean indicating whethere the VAE branch is present or not
    """
    
    model.eval()
    test_bar = tqdm(test_loader, desc=f"[Validation]")
    metrics = np.zeros((3,))
    
    kl_loss_ref = CustomKLLoss() #KL divergence from the github repo
    MSE_loss = nn.MSELoss()
    
    with torch.no_grad():
        for out_img, inp_img, mask in test_bar:
            
            seg_out, rec_y_pred, y_mid = model(inp_img)
            seg_out = torch.round(seg_out)
            dice = soft_dice_coeff(seg_out, mask)
           
            metrics[0] += dice.mean()
                
            if(VAE_enable):
                est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
                metrics[1] += MSE_loss(rec_y_pred, out_img)
                metrics[2] += kl_loss_ref(est_mean, est_std)  
    
    metrics = metrics / len(test_bar)
    
    print(f"--- Validation results --- DICE: {metrics[0]:3f}, MSE: {metrics[1]:3f}, KL {metrics[2]:3f}")
    
    return metrics

def bin_mask_2_multi(mask):
    mask_temp = torch.round(mask)
    num_classes = 3
    multi_mask = torch.zeros(mask.shape[1:3])
    for i in range(num_classes):
        multi_mask[mask_temp[i,:,:]==1] = i
    return multi_mask.squeeze()
    
    
def plot_examples(model, test_dataset, slices, save_path, net_type, VAE_enable = True):
    
    model.eval()
    print("Plotting results" + "-"*60)
    
    kl_loss_ref = CustomKLLoss() #KL divergence from the github repo
    MSE_loss = nn.MSELoss()
    fs = 16 #fontsize
    
    with torch.no_grad():
        j = 0
        for i in slices:
            
            if(i >= len(test_dataset)):
                continue
            
            out_img, inp_img, mask = test_dataset[i]
            
            mask = mask.unsqueeze(0)
            out_img = out_img.unsqueeze(0)
            inp_img = inp_img.unsqueeze(0)
            
            seg_out, rec_pred, _ = model(inp_img)
            seg_out = torch.round(seg_out)
            dice_coeff = soft_dice_coeff(seg_out, mask).squeeze()
            
            
            seg_out = seg_out[0,:,seg_out.shape[2]//2,:,:].squeeze()
            mask = mask[0,:,mask.shape[2]//2,:,:].squeeze()
            HR_image = out_img[0,0,out_img.shape[2]//2,:,:].squeeze()
            degrad_image = inp_img[0,0,inp_img.shape[2]//2,:,:].squeeze()
            
  
            if(VAE_enable):

                mse_loss = MSE_loss(rec_pred, out_img)
                vae_out_2d = rec_pred[0,0,rec_pred.shape[2]//2,:,:].squeeze()

                fig, ax = plt.subplots(1,5,figsize = (17,5))
                plt.gray()
                ax[0].imshow(HR_image.cpu())
                ax[0].set_title("HR central slice", fontsize = fs)
                ax[1].imshow(degrad_image.cpu())
                ax[1].set_title("Degraded central slice", fontsize = fs)
                ax[2].imshow(mask.cpu().permute(1,2,0))
                ax[2].set_title("True mask", fontsize = fs)
                ax[3].imshow(seg_out.cpu().permute(1,2,0))
                ax[3].set_title(f"Pred mask \n Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                ax[4].imshow(vae_out_2d.cpu())
                ax[4].set_title(f"VAE output | MSE = {mse_loss:.3f}", fontsize = fs)

            else:
                fig, ax = plt.subplots(1,4,figsize = (15,5))
                plt.gray()
                ax[0].imshow(HR_image.cpu())
                ax[0].set_title("HR central slice", fontsize = fs)
                ax[1].imshow(degrad_image.cpu())
                ax[1].set_title("Degraded central slice", fontsize = fs)
                ax[2].imshow(mask.cpu().permute(1,2,0))
                ax[2].set_title("HR true mask", fontsize = fs)
                ax[3].imshow(seg_out.cpu().permute(1,2,0))
                ax[3].set_title(f"Pred mask \n Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)


            for ax in fig.get_axes():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='x', length=0)
                ax.tick_params(axis='y', length=0)

            fig.tight_layout()
            fig.savefig(save_path / f"out_{j}.png")
            plt.close(fig)
            j += 1
        
    
    