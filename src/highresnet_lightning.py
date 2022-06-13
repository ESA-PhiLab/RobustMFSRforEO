import itertools
import os
from typing import Iterable, Generator

import numpy as np
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.highresnet import HighResNet, ShiftNet
from src.ssim_loss import SSIMLoss, MultiScaleSSIMLoss, ssim
from src.losses import masked_MSE, cMSE, MSELossFunc
import wandb

array = np.ndarray


class HighResNetLightning(pl.LightningModule):

    def __init__(
        self,
        net_args,
        lr : float = 0.0007,
        lr_decay : float = 0.97,
        lr_patience : int = 3,
        clouds : bool = False,
        shiftnet : bool = False,
        loss : str = 'MSE',
    ) -> None:
        """
        Args:
            net_args : Namespace, configures HighRes-net.
            lr : float, learning rate.
            lr_decay : float, learning rate scheduler decay.
            lr_patience : float, learning rate scheduler patience.
            clouds : bool, include clouds channel in the input.
            shiftnet : bool, register the HighRes-net output with ShiftNet.
            loss : str, loss function to use. One of ['MSE'|'MAE'|'SSIM'|'MSSSIM'|'masked_MSE'|'cMSE'].
        """

        super().__init__()

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.shiftnet = shiftnet
        self.clouds = clouds

        # Initialize Models
        if self.clouds:
            net_args.in_channels += 1
        self.fusion_model = HighResNet(**vars(net_args))

        ## TODO: use RegisteredLoss here ...
        if self.shiftnet:
            self.registration_model = ShiftNet(in_channel=net_args.out_channels)
        else:
            pass

        self.lossname = loss
        if loss == 'MSE':
            self.loss = nn.MSELoss(reduction='none')
        elif loss == 'MAE':
            self.loss = nn.L1Loss(reduction='none')
        elif loss == 'SSIM':
            self.loss = SSIMLoss(reduction='mean')
        elif loss == 'MSSSIM':
            self.loss = MultiScaleSSIMLoss(kernel_size=9, reduction='mean')
        elif loss == 'masked_MSE':
            self.loss = masked_MSE
        elif loss == 'cMSE':
            self.loss = cMSE
        else:
            raise ValueError(f"Value expected for `loss`: ['MSE'|'MAE'|'SSIM'|'MSSSIM'|'masked_MSE'|'cMSE']. Got '{loss}'.")


    def forward(self, item) -> Tensor:
        """
        Args:
            item : dictionary with the following low-res data:
                images : tensor (B, T, C, H, W), low-resolution images.
                revisits_indicator : tensor (B, T), binary indicator of revisits that are paddings.
                    HighRes-net requires the number of low-resolution images to be a power of 2
                    due to the recursive fusion, so `item['images']` might have been padded accordingly.
                clouds : tensor (B, T, H, W), cloud mask data. Not necessarily binary.
        """
        x = self.create_inputs(item)
        r = item['revisits_indicator']
        sr = self.fusion_model(x, r)
        return sr


    def prediction_step(self, item : dict) -> Tensor:
        return self.forward(item)


    def training_step(self, batch : dict, batch_idx : int) -> float:
        """
        Args:
            batch includes:
                lowres : tensor (B, T, C, H, W), low-resolution images.
                lowres_status: tensor (B, 1, H, W), low-res cloud mask. Not necessarily binary.
                revisits_indicator : tensor (B, T), binary indicator of revisits that are paddings.
                    HighRes-net requires the number of low-resolution images to be a power of 2
                    due to the recursive fusion, so `item['images']` might have been padded accordingly.
                highres : tensor (B, C, H, W), high-res image(s).
                highres_status: tensor (B, 1, H, W), high-res cloud map.
        """

        hr_items = batch['highres']
        lr_items = batch['lowres']
        ## TODO: breaks when T > 1
        hr, hr_clouds = hr_items['images'][:, 0], hr_items['clouds'][:, 0] 

        ## Fusion
        lr = self.create_inputs(lr_items)
        r = lr_items['revisits_indicator']
        sr = self.fusion_model(lr, r)

        ## Registration
        if self.shiftnet:
            registration_input = torch.cat([hr, sr], 1)  # Concat on channel dimension
            ## Estimate translation parameters
            shifts = self.registration_model(registration_input)  # (B, 2)
            sr = self.registration_model.transform(shifts, sr)  # Align
            sr = torch.clamp(sr, min=0.0, max=1.0)
        
        ## Metrics
        loss = self.clear_loss(sr, hr, hr_clouds)
        self.log(self.lossname, loss)
        return loss


    def validation_step(self, batch : dict, batch_idx : int) -> None:
        hr_items = batch['highres']
        lr_items = batch['lowres']

        hr = hr_items['images'][:, 0] # (B, C, H, W)
        hr_clouds = hr_items['clouds'][:, 0]  # (B, 1, H, W)

        ## Fusion
        sr = self.forward(lr_items)  # (B, C, H, W)
        if self.shiftnet:
            registration_input = torch.cat([hr, sr], 1)  # Concat on channel dimension
            ## Estimate translation parameters
            shifts = self.registration_model(registration_input)  # (B, 2)
            sr = self.registration_model.transform(shifts, sr)  # Align
            sr = torch.clamp(sr, min=0.0, max=1.0)

        ## Log images
        if batch_idx == 0 and self.logger is not None:
            self.logger.experiment.log({"superres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in sr]})
            self.logger.experiment.log({"highres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in hr]})

        ## Metrics
        mse = F.mse_loss(sr, hr)
        self.log('val_MSE', mse)
        self.log('val_PSNR', - 10 * torch.log10(mse + 1e-8))
        self.log('val_SSIM', ssim(sr, hr))
        
        return {'val_loss': mse}


    def test_step(self, batch : dict, batch_idx : int) -> None:
        hr_items = batch['highres']
        lr_items = batch['lowres']

        hr = hr_items['images'][:, 0]   # (B, C, H, W)
        hr_clouds = hr_items['clouds'][:, 0]   # (B, 1, H, W)

        ## Fusion
        sr = self.forward(lr_items)  # (B, C, H, W)
        if self.shiftnet:
            registration_input = torch.cat([hr, sr], 1)  # Concat on channel dimension
            ## Estimate translation parameters
            shifts = self.registration_model(registration_input)  # (B, 2)
            sr = self.registration_model.transform(shifts, sr)  # Align
            sr = torch.clamp(sr, min=0.0, max=1.0)
        ## Metrics
        mse = F.mse_loss(sr, hr)
        self.log('test_MSE', mse)
        self.log('test_PSNR', - 10 * torch.log10(mse + 1e-8))
        self.log('test_SSIM', ssim(sr, hr))

        return {'test_loss': mse}



    def configure_optimizers(self) -> dict:
        if self.shiftnet:
            optimizer = torch.optim.Adam(list(self.fusion_model.parameters()) + list(self.registration_model.parameters()), self.lr)
        else:
            optimizer = torch.optim.Adam(self.fusion_model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_decay, verbose=True, patience=self.lr_patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_MSE"}


    def patch_iterator(self, img : array, positions : Iterable, size : int) -> Generator[array, None, None]:
        """Iterator across square patches of `img` located in `positions`."""
        for x, y in positions:
            yield self.get_patch(img=img, x=x, y=y, size=size)


    def shift_MSE(self, sr : Tensor, hr : array, hr_map : array, border_w : int = 3) -> float:
        """
        MSE score adjusted for registration errors.
        Computes the min MSE score across shifts of up to `border_w` pixels.

        Args:
            sr: Tensor (C, H, W), super-resolved image.
            hr: Tensor (C, H, W), high-res ground-truth image.
            hr_map: Tensor (1, H, W), high-res status map.
            border_w: int, width of the trimming border around `hr` and `hr_map`.

        Returns:
            float, score of the super-resolved image.
        """

        size = sr.shape[-1] - (2 * border_w)  # patch size
        sr = self.get_patch(img=sr, x=border_w, y=border_w, size=size)
#         sr = sr.unsqueeze(0)
        pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
        iter_hr = self.patch_iterator(img=hr, positions=pos, size=size)
        iter_hr_map = self.patch_iterator(img=hr_map, positions=pos, size=size)

        # TODO: compute various losses on validation data.
#         if self.lossname == 'masked_MSE':
#             loss = masked_MSE
#         if self.lossname == 'cMSE':
#             loss = cMSE
        if self.lossname == 'MSE':
            loss = MSELossFunc

        shift_loss = np.array([loss(sr, hr, hrm) for hr, hrm in zip(iter_hr, iter_hr_map)])

        return np.min(shift_loss, axis=0)


    def get_patch(self, img : array, x : int, y : int, size : int = 128) -> array:
        """
        Slices out a square patch from `img` starting from the (x,y) top-left corner.
        If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
        and the output has shape (l, size, size).
        Args:
            img: numpy.ndarray or Tensor (..., H, W), input image.
            x, y: int, top-left corner of the patch.
            size: int, patch size.
        Returns:
            numpy.ndarray or Tensor (..., size, size)
        """
        return img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays


    def create_inputs(self, items : dict) -> Tensor:
        """
        Creates input for forward pass based on options.
        Args:
            items: dict, lowres batch.
        """
        if self.clouds:
            return torch.cat([items['images'], items['clouds']], 2)
        else:
            return items['images']


    def clear_loss(self, sr : Tensor, hr : Tensor, hr_maps : Tensor) -> Tensor:
        if 'SSIM' in self.lossname:
            # SSIM would be affected by Zero-ed out pixels and pixel neighbourhood.
            hr_maps = (hr_maps==0).float()
            return self.loss(sr * hr_maps, hr * hr_maps)
#         elif self.lossname == 'masked_MSE':
#             return torch.sum(self.loss(sr, hr, hr_maps))
#         elif self.lossname == 'cMSE':
#             return torch.sum(self.loss(sr, hr, hr_maps))
        else:
            nclear = hr_maps.numel() - torch.sum(hr_maps)
            hr_maps = (hr_maps==0).float()
            return torch.sum(self.loss(sr * hr_maps, hr * hr_maps)) / nclear
