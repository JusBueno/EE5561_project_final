import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def soft_dice_coeff(pred, target, epsilon=1e-8):
    """
    Returns soft dice coefficient along batches and channels
    """
    if(pred.ndim == 4): #[B, C, H, W]
        intersection = (pred * target).sum(dim=(2, 3))
        pred_sq_sum = (pred ** 2).sum(dim=(2, 3))
        target_sq_sum = (target ** 2).sum(dim=(2, 3))
        dice = (2 * intersection + epsilon) / (pred_sq_sum + target_sq_sum + epsilon)
        
    elif(pred.ndim == 5): #[B, C, D, H, W]
        intersection = (pred * target).sum(dim=(2, 3, 4))
        pred_sq_sum = (pred ** 2).sum(dim=(2, 3, 4))
        target_sq_sum = (target ** 2).sum(dim=(2, 3, 4))
        dice = (2 * intersection + epsilon) / (pred_sq_sum + target_sq_sum + epsilon)

    return dice  # shape (B, C)


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division, 
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        
#         intersection = torch.sum(torch.mul(y_pred, y_true)) 
#         union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps
#         dice = 2 * intersection / union 
#         dice_loss = 1 - dice

        dice_loss = 1-soft_dice_coeff(y_pred, y_true)
    
        return dice_loss.sum(dim=1).mean()
    


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    Accepts either 5 inputs (if using VAE) or 2 (if not using VAE)
    '''
    def __init__(self, k1=0.1, k2=0.1,VAE_enable=True, separate = False):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()
        self.VAE_enable = VAE_enable
        self.separate = separate


    def forward(self, seg_y_pred, seg_y_true, rec_y_pred = None, rec_y_true = None, y_mid = None,):
        dice_loss = self.dice_loss(seg_y_pred, seg_y_true)
        if(not self.VAE_enable):
            return dice_loss, dice_loss, 0, 0
        
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        l2_loss = self.l2_loss(rec_y_pred, rec_y_true)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        
        if(self.separate):
            return combined_loss, dice_loss, l2_loss, kl_div 

        return combined_loss
    
    
#Previous functions
# def kl_loss(mu, logvar):
#     var = torch.exp(logvar)
#     kl = (mu**2 + var - logvar - 1).sum(dim=1).mean()
#     return kl

# MSE_loss = nn.MSELoss()
# dice_loss = SoftDiceLoss()

# def combined_loss(seg_out, seg_target, vae_out, vae_target, mu, logvar, w1 = 0.1, w2 = 0.1):
#     lossSD = dice_loss(seg_out, seg_target) 
#     lossL2 = MSE_loss(vae_out, vae_target)
#     lossKL = kl_loss(mu, logvar)  
#     return lossSD + w1 * lossL2 + w2 * lossKL



