from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.criterion import *


def test_model(model, test_loader, VAE_enable = True, UNET_enable = True):
    """
    Testing function for the model
    Input:
        model: model to be tested
        test_loader: data loader for the test dataset
        VAE_enable (bool): indicating whethere the VAE decoder branch is present or not
        UNET_enable (bool): indicating whethere the UNET decoder branch is present or not
    Output:
        metrics: 3x1 np array including the Dice coefficient, L2 loss and KL divergence loss
        averaged over the test dataset. In case of not using the VAE or UNET decoder branch,
        whatever loss that does not apply is set to 0
    """
    
    model.eval()
    test_bar = tqdm(test_loader, desc=f"[Validation]")
    metrics = np.zeros((3,))
    kl_loss_ref = CustomKLLoss()
    MSE_loss = nn.MSELoss()
    dice_loss = SoftDiceLoss()
    with torch.no_grad():
        for out_img, inp_img, mask in test_bar:
            
            seg_out, rec_y_pred, y_mid = model(inp_img)

            if(UNET_enable):
                seg_out = torch.round(seg_out)
                metrics[0] += dice_loss(seg_out, mask)
                
            if(VAE_enable):
                est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
                metrics[1] += MSE_loss(rec_y_pred, out_img)
                metrics[2] += kl_loss_ref(est_mean, est_std)  
    
    metrics = metrics / len(test_bar)
    
    print(f"--- Validation results --- DICE loss: {metrics[0]:3f}, MSE: {metrics[1]:3f}, KL {metrics[2]:3f}")
    
    return metrics


    
def plot_examples(model, test_dataset, slices, save_path, VAE_enable = True, UNET_enable = True, threeD = True):
    """
    Plot segmentation and reconstruction results for different variation of the network.
    The size and number of figures plotted depend on the VAE_enable and UNET_enable parameters
        Input:
        model: model to be tested
        test_loader: data loader for the test dataset
        VAE_enable (bool): indicating whethere the VAE decoder branch is present or not
        threeD (bool): indicating whethere we use 3D data or 2D data
    """
    
    model.eval()
    print("Plotting results" + "-"*60)
    
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

            if not threeD:
                mask = mask.unsqueeze(2)
                seg_out = seg_out.unsqueeze(2)
                inp_img = inp_img.view(1, 4, inp_img.shape[1]//4, inp_img.shape[2], inp_img.shape[3])
                out_img = out_img.view(1, 4, out_img.shape[1]//4, out_img.shape[2], out_img.shape[3])
                if VAE_enable:
                    rec_pred = rec_pred.view(1, 4, rec_pred.shape[1]//4, rec_pred.shape[2], rec_pred.shape[3])
            
            HR_image = out_img[0,0,out_img.shape[2]//2,:,:].squeeze()
            degrad_image = inp_img[0,0,inp_img.shape[2]//2,:,:].squeeze()

            if(UNET_enable and not VAE_enable):
                seg_out = torch.round(seg_out)
                m = seg_out.shape[2]//2
                seg_out_2d = seg_out[0:1,:,m:m+1,:,:]
                mask_2d = mask[0:1,:,m:m+1,:,:]

                dice_coeff = soft_dice_coeff(seg_out_2d, mask_2d).squeeze()
                
                seg_out_2d = seg_out_2d.squeeze()
                mask_2d = mask_2d.squeeze()

                fig, ax = plt.subplots(1,4,figsize = (13.5,3.5))
                plt.gray()
                ax[0].imshow(HR_image.cpu())
                ax[0].set_title("HR central slice", fontsize = fs)
                ax[1].imshow(degrad_image.cpu())
                ax[1].set_title("Degraded central slice", fontsize = fs)
                ax[2].imshow(mask_2d.cpu().permute(1,2,0))
                ax[2].set_title("HR true mask", fontsize = fs)
                ax[3].imshow(seg_out_2d.cpu().permute(1,2,0))
                ax[3].set_title(f"Pred mask \n Dice = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)

            if(VAE_enable and not UNET_enable):

                mse_loss = MSE_loss(rec_pred, out_img)
                vae_out_2d = rec_pred[0,0,rec_pred.shape[2]//2,:,:].squeeze()

                fig, ax = plt.subplots(1,3,figsize = (10,3.5))
                plt.gray()
                ax[0].imshow(HR_image.cpu())
                ax[0].set_title("HR central slice", fontsize = fs)
                ax[1].imshow(degrad_image.cpu())
                ax[1].set_title("Degraded central slice", fontsize = fs)
                ax[2].imshow(vae_out_2d.cpu())
                ax[2].set_title(f"VAE output | MSE = {mse_loss:.3f}", fontsize = fs)

            if(VAE_enable and UNET_enable):

                seg_out = torch.round(seg_out)
                m = seg_out.shape[2]//2
                seg_out_2d = seg_out[0:1,:,m:m+1,:,:]
                mask_2d = mask[0:1,:,m:m+1,:,:]

                dice_coeff = soft_dice_coeff(seg_out_2d, mask_2d).squeeze()
                
                seg_out_2d = seg_out_2d.squeeze()
                mask_2d = mask_2d.squeeze()

                mse_loss = MSE_loss(rec_pred, out_img)
                vae_out_2d = rec_pred[0,0,rec_pred.shape[2]//2,:,:].squeeze()

                fig, ax = plt.subplots(1,5,figsize = (17,3.5))
                plt.gray()
                ax[0].imshow(HR_image.cpu())
                ax[0].set_title("HR central slice", fontsize = fs)
                ax[1].imshow(degrad_image.cpu())
                ax[1].set_title("Degraded central slice", fontsize = fs)
                ax[2].imshow(mask_2d.cpu().permute(1,2,0))
                ax[2].set_title("True mask", fontsize = fs)
                ax[3].imshow(seg_out_2d.cpu().permute(1,2,0))
                ax[3].set_title(f"Pred mask \n Dice coeff = {dice_coeff[0]:.2f}, {dice_coeff[1]:.2f}, {dice_coeff[2]:.2f}", fontsize = fs)
                ax[4].imshow(vae_out_2d.cpu())
                ax[4].set_title(f"VAE output | MSE = {mse_loss:.3f}", fontsize = fs)


            for ax in fig.get_axes():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='x', length=0)
                ax.tick_params(axis='y', length=0)

            fig.tight_layout()
            fig.savefig(save_path / f"out_{j}.png")
            plt.close("all")
            j += 1
        

def plot_loss_curves(results_path, validation_metrics, training_metrics, epoch, VAE_enable, UNET_enable, net_type):
    """
    Plot training and validation loss curves after training.
    """

    best_val_dice = 1-validation_metrics[:epoch+1, 0].min(axis=0)

    if VAE_enable and UNET_enable:
        fig,  ax = plt.subplots(1,3,figsize = (8,3.5))
        title_list = ["Dice coefficient", "MSE", "KL Divergence"]
        
        ax[0].plot(training_metrics[:epoch+1, 0], label='Training')
        ax[0].plot(validation_metrics[:epoch+1, 0], label='Validation')
        ax[0].set_xlabel("Epochs", fontsize = 13)
        ax[0].set_ylabel("Dice loss", fontsize = 13)
        ax[0].grid()
        
        for i in range(1,3):
            ax[i].plot(training_metrics[:epoch+1, i], label='Training')
            ax[i].plot(validation_metrics[:epoch+1, i], label='Validation')
            ax[i].set_xlabel("Epochs", fontsize = 13)
            ax[i].set_ylabel(title_list[i], fontsize = 13)
            ax[i].grid()
        
        ax[2].legend(fontsize = 10)
        fig.suptitle(f"Network: {net_type}, VAE is {VAE_enable},\nbest DICE coeff = {best_val_dice:.3f}", fontsize=13)
    
    if VAE_enable and not UNET_enable:
        fig,  ax = plt.subplots(1,2,figsize = (6,3.5))
        title_list = ["Dice coefficient", "MSE", "KL Divergence"]
        
        for i in range(1,3):
            ax[i-1].plot(training_metrics[:epoch+1, i], label='Training')
            ax[i-1].plot(validation_metrics[:epoch+1, i], label='Validation')
            ax[i-1].set_xlabel("Epochs", fontsize = 13)
            ax[i-1].set_ylabel(title_list[i], fontsize = 13)
            ax[i-1].grid()
        
        ax[1].legend(fontsize = 10)
        
        
    elif UNET_enable and not VAE_enable:
        fig,  ax = plt.subplots(1,1,figsize = (3.5,3.5))
        
        ax.plot(training_metrics[:epoch+1, 0], label='Training')
        ax.plot(validation_metrics[:epoch+1, 0], label='Validation')
        ax.set_xlabel("Epochs", fontsize = 13)
        ax.set_ylabel("Dice loss", fontsize = 13)
        ax.legend(fontsize = 10)
        ax.set_title(f"Network: {net_type}, VAE is {VAE_enable}, \nbest DICE coeff = {best_val_dice:.3f}", fontsize=13)
        ax.grid()

    fig.tight_layout()
    fig.savefig(results_path/"loss_curves.png")
    plt.close("all")