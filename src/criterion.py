import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from config import Configs, save_configs

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
        
        intersection = torch.sum(torch.mul(y_pred, y_true)) 
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps
        dice = 2 * intersection / union 
        dice_loss = 1 - dice
        return dice_loss

        #dice_loss = 1-soft_dice_coeff(y_pred, y_true)
        #return dice_loss.sum(dim=1).mean()
    


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1


class CustomKLLoss_2(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss_2, self).__init__()

    def forward(self, mean, logvar):
        var = torch.exp(logvar)
        return torch.mean(torch.mul(mean, mean)) + torch.mean(var) - torch.mean(logvar) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    Accepts either 5 inputs (if using VAE) or 2 (if not using VAE)
    '''
    def __init__(self, params, separate = True):
        super(CombinedLoss, self).__init__()
        self.params = params
        self.separate = separate
        
        if self.params.VAE_warmup:
            self.kl_annealer = Annealer(17, start_epochs=2)
            self.recon_annealer = Annealer(5, baseline=0.2)
            self.kl_annealer.current_step = params.start_epoch
            self.recon_annealer.current_step = params.start_epoch
        else: self.kl_annealer = self.recon_annealer = None

        if self.params.logvar_out:
            self.kl_loss = CustomKLLoss_2()
        else:
            self.kl_loss = CustomKLLoss()

        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()


    def forward(self, seg_y_pred, seg_y_true, rec_y_pred = None, rec_y_true = None, y_mid = None):

        dice_loss = self.dice_loss(seg_y_pred, seg_y_true)

        if(self.params.VAE_enable):
            est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
            l2_loss = self.l2_loss(rec_y_pred, rec_y_true)
            kl_div = self.kl_loss(est_mean, est_std)
            if(self.kl_annealer is not None):
                kl_div = self.kl_annealer(kl_div)
            if(self.recon_annealer is not None):
                l2_loss = self.recon_annealer(l2_loss)

        else: l2_loss = kl_div = 0

        combined_loss = dice_loss + self.params.k1 * l2_loss + self.params.k2 * kl_div
        
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


##VAE Warmup (also called annealing) test

import math


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape='linear', baseline=0.0, cyclical=False, disable=False, start_epochs = 0):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """

        self.current_step = 0
        self.start_epochs = start_epochs

        if shape not in ['linear', 'cosine', 'logistic']:
            raise ValueError("Shape must be one of 'linear', 'cosine', or 'logistic.")
        self.shape = shape

        if not 0 <= float(baseline) <= 1:
            raise ValueError("Baseline must be a float between 0 and 1.")
        self.baseline = baseline

        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("Argument total_steps must be an integer greater than 0")
        self.total_steps = total_steps

        if type(cyclical) is not bool:
            raise ValueError("Argument cyclical must be a boolean.")
        self.cyclical = cyclical

        if type(disable) is not bool:
            raise ValueError("Argument disable must be a boolean.")
        self.disable = disable

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        if self.disable:
            return kld
        out = kld * self._slope()
        return out

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def set_cyclical(self, value):
        if not isinstance(value, bool):
            raise ValueError("Argument to cyclical method must be a boolean.")
        self.cyclical = value
        return


    def _slope(self):

        if self.current_step < self.start_epochs:
            return 0
        
        if self.shape == 'linear':
            y = ((self.current_step-self.start_epochs) / (self.total_steps-self.start_epochs))
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * ((self.current_step-self.start_epochs) / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - (self.current_step-self.start_epochs))
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1.0
        y = self._add_baseline(y)
        return y

    def _add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out