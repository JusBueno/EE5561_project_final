import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return (1-soft_dice_coeff(pred, target, self.epsilon)).sum(dim = 1).mean() #Sum loss channels
    
def soft_dice_coeff(pred, target, epsilon=1e-6):
    if(pred.ndim == 4):
        intersection = (pred * target).sum(dim=(2, 3))
        denominator = (pred ** 2).sum(dim=(2, 3)) + (target ** 2).sum(dim=(2, 3))
        dice = (2. * intersection) / (denominator + epsilon)
        
    else:
        intersection = (pred * target).sum(dim=(-1, -2))        # (B, C, D)
        pred_sum     = pred.sum(dim=(-1, -2))                    # (B, C, D)
        target_sum   = target.sum(dim=(-1, -2))                  # (B, C, D)
        dice_per_depth = (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
        dice = dice_per_depth.mean(dim=-1)                  # (B, C)
    
    return dice #Output size = [num_batches, num_channels]

def kl_loss(mu, logvar):
    var = torch.exp(logvar)
    kl = (mu**2 + var - logvar - 1).sum(dim=1).mean()
    return kl

MSE_loss = nn.MSELoss()
dice_loss = SoftDiceLoss()

def combined_loss(seg_out, seg_target, vae_out, vae_target, mu, logvar, w1 = 0.1, w2 = 0.1):
    
    lossSD = dice_loss(seg_out, seg_target) 
    lossL2 = MSE_loss(vae_out, vae_target)
    lossKL = kl_loss(mu, logvar)
        
    return lossSD + w1 * lossL2 + w2 * lossKL