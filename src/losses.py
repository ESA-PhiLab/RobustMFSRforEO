from math import exp

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

## GAN Losses
import torchvision
from torchvision.models import vgg19
from torchvision.transforms.functional import normalize

from src.datasources import S2_BANDS
from src.datasets import SN7Dataset



def cMSE(sr : Tensor, hr : Tensor, hr_maps : Tensor) -> torch.float32:
    """
    Clear-MSE per batch instance.
    See also, ESA Loss: https://kelvins.esa.int/proba-v-super-resolution/scoring/

    Args:
        sr: tensor (B, C, H, W), super-resolved images.
        hr: tensor (B, C, H, W), high-res images.
        hr_maps: tensor (B, 1, H, W), high-res status maps.

    Returns:
        Mean cMSE across batch.
    """
    mse = nn.MSELoss(reduction='none')
    hr_maps = (hr_maps == 0).float()
    nclear = torch.sum(hr_maps, dim=(-3, -2, -1), keepdims=True)  + 1e-5  # Number of clear pixels in target image
    bright_inter = torch.sum(hr_maps * (hr - sr), dim=(-2, -1), keepdims=True)  # .clone().detach()
    ## TODO: investigate further
    bright =  bright_inter / nclear  # Correct for brightness per channel
    ## cMSE(A,B) per instance.
    loss = torch.sum(hr_maps * mse(sr + bright, hr), dim=(-3, -2, -1)) / nclear
    return loss.mean()


def cPSNR(sr : Tensor, hr : Tensor, hr_maps : Tensor) -> Tensor:
    """
    Clear Peak Signal-to-Noise Ratio. The PSNR score, adjusted for brightness and other volatile features, e.g. clouds.
    Args:
        sr: tensor (B, H, W), super-resolved images.
        hr: tensor (B, H, W), high-res images.
        hr_maps: tensor (B, H, W), high-res status maps.
    Returns:
        tensor (B), metric for each super-resolved image.
    """
    cmse = cMSE(sr, hr, hr_maps)
    return -10 * torch.log10(cmse)


def masked_MSE(sr : Tensor, hr : Tensor, hr_maps : Tensor) -> Tensor:
    """
    Computes MSE loss for each instance in a batch, excluding clouds.
    Args:
        srs: tensor (B, C, H, W), super-resolved images.
        hrs: tensor (B, C, H, W), high-res images.
        hr_maps: tensor (B, C, H, W), high-res status maps.
    Returns:
        tensor (B,), metric for each super-resolved image.
    """
    criterion = nn.MSELoss(reduction='none')
    hr_maps = (hr_maps==0).float()
    nclear = hr_maps.numel() - torch.sum(hr_maps)
    loss = criterion(hr_maps * sr, hr_maps * hr)
    return torch.sum(loss, dim=(-3, -2, -1)) / nclear


def masked_PSNR(srs, hrs, hr_maps):
    """
    Computes MSE loss for each instance in a batch, excluding clouds.
    Args:
        srs: tensor (B, W, H), super resolved images
        hrs: tensor (B, W, H), high-res images
        hr_maps: tensor (B, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    criterion = nn.MSELoss(reduction='none')
    hr_maps = (hr_maps==0).float()
    loss = criterion(hr_maps * srs, hr_maps * hrs)
    loss = torch.mean(loss, dim=(-3,-2, -1))
    return -10 * torch.log10(loss)  # PSNR


def MSELossFunc(srs, hrs, hr_maps):
    """
    Computes MSE loss for each instance in a batch, excluding clouds.
    Args:
        srs: tensor (B, C, W, H), super resolved images
        hrs: tensor (B, C, W, H), high-res images
        hr_maps: tensor (B, C, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    criterion = nn.MSELoss(reduction='none')
    loss = criterion(srs, hrs)
    return torch.mean(loss)#, dim=(-4,-3,-2, -1))


def MAELossFunc(srs, hrs, hr_maps):
    """
    Computes MAE loss for each instance in a batch, excluding clouds.
    Args:
        srs: tensor (B, C, W, H), super resolved images
        hrs: tensor (B, C, W, H), high-res images
        hr_maps: tensor (B, C, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    loss = F.l1_loss(srs, hrs, reduction='none')
    return torch.mean(loss)#, dim=(-4,-3,-2, -1))



class ContentLoss(nn.Module):
    # Add pixel-wise loss
    def __init__(self):
        super(ContentLoss, self).__init__()
        
        # Sentinel Standardization
        self.sentinel_mean, self.sentinel_std = SN7Dataset.mean['sentinel'][S2_BANDS['true_color']], SN7Dataset.std['sentinel'][S2_BANDS['true_color']]
        self.max95 = torch.as_tensor(self.sentinel_mean + 2*self.sentinel_std)
        # Planet Standardization
        self.planet_mean, self.planet_std = SN7Dataset.mean['planet'], SN7Dataset.std['planet']

        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:36]).eval()

        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, superres, highres):
#         B, C, H, W = lowres.shape
#         lowres = torch.stack([lowres[:,0,...]*self.max95[0], lowres[:,1,...]*self.max95[1], lowres[:,2,...]*self.max95[2]], dim=1)
#         input_tensor = torchvision.transforms.functional.normalize(lowres, mean=self.sentinel_mean, std=self.sentinel_std)
#         input_tensor.view(B,C,H,W)
        superres = superres * 255
        input_tensor = torchvision.transforms.functional.normalize(superres, mean=self.planet_mean, std=self.planet_std)
        
        highres = highres * 255
        target_tensor = torchvision.transforms.functional.normalize(highres, mean=self.planet_mean, std=self.planet_std)
        
        return F.mse_loss(
            self.features(input_tensor),
            self.features(target_tensor)
        )



class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, discriminator_fake_outputs):
        return - discriminator_fake_outputs.log().sum()
