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
        for out_imgs, inp_imgs, mask in test_bar:
           
            
#             if net_type == "VAE_2D":
#                 central_index = out_imgs.shape[1]//2
#                 central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output
#                 seg_out, vae_out, mu, logvar = model(inp_imgs)
#                 seg_out = torch.round(seg_out)
#                 metrics[0] += soft_dice_coeff(seg_out, mask).mean()
#                 metrics[1] += MSE_loss(vae_out, central_slice)
#                 metrics[2] += kl_loss(mu, logvar)
                
#             elif net_type == "UNET_2D":
#                 central_index = out_imgs.shape[1]//2
#                 central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output
#                 seg_out = model(inp_imgs)
#                 metrics[0] += soft_dice_coeff(seg_out, mask).mean()

#             elif net_type in ["REF", 
                
            if(VAE_enable):
                seg_out, rec_y_pred, y_mid = model(inp_imgs)
                seg_out = torch.round(seg_out)
                est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
                dice = soft_dice_coeff(seg_out, mask)

                metrics[0] += dice.mean()
                metrics[1] += MSE_loss(rec_y_pred, out_imgs)
                metrics[2] += kl_loss_ref(est_mean, est_std)

            else:
                seg_out, _, _ = model(inp_imgs)
                seg_out = torch.round(seg_out)
                dice = soft_dice_coeff(seg_out, mask)
                metrics[0] += dice.mean()

    
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
    
    with torch.no_grad():
        j = 0
        for i in slices:
            
            out_img, inp_img, mask = test_dataset[i]
            
            mask = mask.unsqueeze(0)
            out_img = out_img.unsqueeze(0)
            inp_img = inp_img.unsqueeze(0)
        
#             if net_type == "VAE_2D":

#                 seg_out, vae_out, mu, logvar = model(inp_img)
#                 seg_out = torch.round(seg_out)
                
#                 central_index = out_img.shape[1]//2
#                 central_slice = out_img[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output

#                 dice_coeff = soft_dice_coeff(seg_out, mask).squeeze()
#                 mse_loss = MSE_loss(vae_out, central_slice)
                
#                 seg_out = seg_out.squeeze().permute(1,2,0)
#                 mask = mask.squeeze().permute(1,2,0)


#                 fig, ax = plt.subplots(2,2,figsize = (10,5))
#                 plt.gray()
#                 ax[0,0].imshow(central_slice.cpu().squeeze())
#                 ax[0,0].set_title("HR central slice")
#                 ax[0,1].imshow(mask.cpu())
#                 ax[0,1].set_title("HR true mask")
#                 ax[1,0].imshow(seg_out.cpu())
#                 ax[1,0].set_title(f"Pred mask | Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}")
#                 ax[1,1].imshow(vae_out.squeeze().cpu())
#                 ax[1,1].set_title(f"VAE output | MSE = {mse_loss:.3f}")

#             elif net_type == "UNET_2D":

#                 seg_out = model(inp_img)
#                 seg_out = torch.round(seg_out)
#                 dice_coeff = soft_dice_coeff(seg_out, mask).squeeze()

#                 central_index = out_img.shape[1]//2
#                 central_slice = out_img[:,central_index,:,:] #Get central slice for plotting

#                 seg_out = seg_out.squeeze().permute(1,2,0)
#                 mask = mask.squeeze().permute(1,2,0)

#                 fig, ax = plt.subplots(1,3,figsize = (10,5))
#                 plt.gray()
#                 ax[0].imshow(central_slice.cpu().squeeze())
#                 ax[0].set_title("HR central slice")
#                 ax[1].imshow(mask.cpu())
#                 ax[1].set_title("HR true mask")
#                 ax[2].imshow(seg_out.cpu())
#                 ax[2].set_title(f"Pred mask | Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}")

#             elif net_type == "ref_3D":
                
            if(VAE_enable):
                seg_pred, rec_pred, _ = model(inp_img)

                seg_pred = torch.round(seg_pred)

                dice_coeff = soft_dice_coeff(seg_pred, mask).squeeze()

                mse_loss = MSE_loss(rec_pred, out_img)
                idx = seg_pred.shape[2]//2

                seg_pred = seg_pred[0,:,idx,:,:].squeeze()
                mask = mask[0,:,idx,:,:].squeeze()
                vae_out_2d = rec_pred[0,0,idx,:,:].squeeze()
                input_2d = out_img[0,0,idx,:,:].squeeze()

                fig, ax = plt.subplots(2,2,figsize = (10,5))
                plt.gray()
                ax[0,0].imshow(input_2d.cpu())
                ax[0,0].set_title("HR central slice")
                ax[0,1].imshow(mask.cpu().permute(1,2,0))
                ax[0,1].set_title("HR true mask")
                ax[1,0].imshow(seg_pred.cpu().permute(1,2,0))
                ax[1,0].set_title(f"Pred mask | Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}")
                ax[1,1].imshow(vae_out_2d.cpu())
                ax[1,1].set_title(f"VAE output | MSE = {mse_loss:.3f}")

            else:
                seg_pred, _, _ = model(inp_img)

                seg_pred = torch.round(seg_pred)

                dice_coeff = soft_dice_coeff(seg_pred, mask).squeeze()

                idx = seg_pred.shape[2]//2

                seg_pred = seg_pred[0,:,idx,:,:].squeeze()
                mask = mask[0,:,idx,:,:].squeeze()
                input_2d = out_img[0,0,idx,:,:].squeeze()

                fig, ax = plt.subplots(1,3,figsize = (10,5))
                plt.gray()
                ax[0].imshow(input_2d.cpu())
                ax[0].set_title("HR central slice")
                ax[1].imshow(mask.cpu().permute(1,2,0))
                ax[1].set_title("HR true mask")
                ax[2].imshow(seg_pred.cpu().permute(1,2,0))
                ax[2].set_title(f"Pred mask | Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}")



            for ax in fig.get_axes():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='x', length=0)
                ax.tick_params(axis='y', length=0)

            fig.tight_layout()
            fig.savefig(save_path / f"out_{j}.png")
            plt.close(fig)
            j += 1
        
    
    