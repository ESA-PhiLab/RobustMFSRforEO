import math
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from src.ssim_loss import SSIMLoss, MultiScaleSSIMLoss, ssim
from src.losses import masked_MSE, cMSE, MSELossFunc
from src.highresnet import ShiftNet
import numpy as np
import torch, itertools
import wandb
from torch.nn import functional as F
import torch

class FSRCNNModule(pl.LightningModule):
    def __init__(self,  net_args, model='FSRCNN', lr=0.0007, lr_decay=0.97, lr_patience=3, shiftnet=False, loss='MSE', registered_val=True):
        super(FSRCNNModule, self).__init__()
        
        if model == 'FSRCNN':
            self.net = FSRCNN(upscale_factor=net_args.upscale_factor, in_channels=net_args.in_channels,
                              out_channels=net_args.out_channels, additional_scaling=net_args.additional_scaling)
        elif model == 'RDN':
            self.net = RDN(upscale_factor=net_args.upscale_factor, in_channels=net_args.in_channels,
                              out_channels=net_args.out_channels, additional_scaling=net_args.additional_scaling)
        else:
            raise NotImplemented
        
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.shiftnet = shiftnet
        if self.shiftnet:
            self.registration_model = ShiftNet(in_channel=net_args.out_channels)
        
        # Choose Loss
        self.lossname = loss
        if loss == 'MSE':
            self.loss = nn.MSELoss(reduction='none')
        elif loss == 'MAE':
            self.loss = nn.L1Loss(reduction='none')
        elif loss == 'SSIM':
            self.loss = SSIMLoss(reduction='mean')
        elif loss == 'MSSSIM':
            self.loss = MultiScaleSSIMLoss(kernel_size=9, reduction='mean')
        else:
            raise NotImplementedError

    def forward(self, lowres_images):
        return self.net(lowres_images)

    def prediction_step(self, lowres_dict):
        indices = (lowres_dict['clearances']).argmax(1)
        lowres_images = lowres_dict['images'][torch.arange(len(indices)), indices, ...]
        return self.net(lowres_images)

    def training_step(self, batch,  batch_idx):
        highres = batch['highres']
        lowres = batch['lowres']
        
        indices = (lowres['clearances']).argmax(1)
        lowres_images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
        highres_images = highres['images'][:, 0] 
        
        superres = self.forward(lowres_images)
        
        if self.shiftnet:
            # register
            # thetas: tensor (B, 2), translation params
            registration_input = torch.cat([highres_images, superres], 1) # concat on channel dimension
            shifts = self.registration_model(registration_input)
            # do registration
            superres = self.registration_model.transform(shifts, superres)
            superres = torch.clamp(superres, min=0.0, max=1.0) 
        
        loss = self.clear_loss(superres, highres_images, highres['clouds'][:, 0])
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        highres = batch['highres']
        lowres = batch['lowres']
        
        indices = (batch['lowres']['clearances']).argmax(1)
        lowres_images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
        highres_images = highres['images'][:, 0] 
        
        superres = self.forward(lowres_images)
        if self.shiftnet:
            # register
            # thetas: tensor (B, 2), translation params
            registration_input = torch.cat([highres_images, superres], 1) # concat on channel dimension
            shifts = self.registration_model(registration_input)
            # do registration
            superres = self.registration_model.transform(shifts, superres)
            superres = torch.clamp(superres, min=0.0, max=1.0)
            
        # calculate metrics
        mse = F.mse_loss(superres, highres_images)
        self.log('val_PSNR', - 10 * torch.log10(mse + 1e-8))
        self.log('val_SSIM', ssim(superres, highres_images))
        self.log('val_MSE', mse)
        
        if batch_idx == 0 and self.logger is not None:
            self.logger.experiment.log({"superres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in superres]})
            self.logger.experiment.log({"highres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in highres_images]})
        return {'val_loss': mse}


    def configure_optimizers(self):
        if self.shiftnet:
            optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.registration_model.parameters()), self.lr)
        else:
            optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_decay, verbose=True, patience=self.lr_patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def shift_loss(self, superres, highres, highres_status, border_w=3):
        """
        MSE score adjusted for registration errors. Computes the min MSE score across shifts of up to `border_w` pixels.
        Args:
            sr: np.ndarray (n, m), super-resolved image
            hr: np.ndarray (n, m), high-res ground-truth image
            hr_map: np.ndarray (n, m), high-res status map
            border_w: int, width of the trimming border around `hr` and `hr_map`
        Returns:
            max_cPSNR: float, score of the super-resolved image
        """
        size = superres.shape[1] - (2 * border_w)  # patch size
        superres = self.get_patch(img=superres, x=border_w, y=border_w, size=size)
        superres = superres.unsqueeze(0)
        pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
        iter_highres = self.patch_iterator(img=highres, positions=pos, size=size)
        iter_highres_status = self.patch_iterator(img=highres_status, positions=pos, size=size)
        
        # TODO: compute various losses on validation data.
        if (self.lossname == 'masked_MSE'):
            site_MSE = np.array([masked_MSE(superres, hr, hrm) for hr, hrm in zip(iter_highres, iter_highres_status)])
        elif (self.lossname == 'MSE'):
            site_MSE = np.array([MSELossFunc(superres, hr, hrm) for hr, hrm in zip(iter_highres, iter_highres_status)])
        else:
            site_MSE = np.array([cMSE(superres, hr, hrm) for hr, hrm in zip(iter_highres, iter_highres_status)])
        min_MSE = np.min(site_MSE, axis=0)
        return min_MSE

    def get_patch(self, img, x, y, size=128):
        """
        Slices out a square patch from `img` starting from the (x,y) top-left corner.
        If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
        and the output has shape (l, size, size).
        Args:
            img: numpy.ndarray (n, m), input image
            x, y: int, top-left corner of the patch
            size: int, patch size
        Returns:
            patch: numpy.ndarray (size, size)
        """

        patch = img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays
        return patch
    
    def patch_iterator(self, img, positions, size):
        """Iterator across square patches of `img` located in `positions`."""
        for x, y in positions:
            yield self.get_patch(img=img, x=x, y=y, size=size)
    
    def clear_loss(self, srs, hrs, hr_maps):
        if 'SSIM' in self.lossname:
            # SSIM would be affected by Zero-ed out pixels and pixel neighbourhood.
            hr_maps = (hr_maps==0).float()
            return self.loss(srs*hr_maps, hrs*hr_maps)
        else:
            nclear = hr_maps.numel() - torch.sum(hr_maps)
            hr_maps = (hr_maps==0).float()
            return torch.sum(self.loss(srs*hr_maps, hrs*hr_maps))/nclear

class FSRCNN(nn.Module):
    r"""Implement the model file in FSRCNN.
    `"Accelerating the Super-Resolution Convolutional Neural Network" <https://arxiv.org/pdf/1608.00367.pdf>`_
    """
    # Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 (the "License");
    #   you may not use this file except in compliance with the License.
    #   You may obtain a copy of the License at
    #
    #       http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    def __init__(self, upscale_factor=2, additional_scaling=None, in_channels=3, out_channels=3):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        # Channel shrinking
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # Channel mapping
        self.map = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Channel expanding
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # Deconvolution
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=upscale_factor,
                                         padding=4, output_padding=upscale_factor - 1)
    
        self.scale = False # do not do additional scaling by default
        if additional_scaling is not None:
            self.scale = True
            self.additional_scaling = nn.Upsample(mode='bicubic', scale_factor=additional_scaling, align_corners=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, lowres):
        out = self.features(lowres)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        
        if self.scale:
            out = self.additional_scaling(out)
        out = self.sigmoid(out)
        return out

# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# https://github.com/thstkdgus35/EDSR-PyTorch/blob/9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2/src/model/rdn.py

import torch
import torch.nn as nn

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, out_channels=3, additional_scaling=None):
        super(RDN, self).__init__()
        r = upscale_factor
        G0 = 64
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }['A']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_channels, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
        
        self.scale = False # do not do additional scaling by default
        if additional_scaling is not None:
            self.scale = True
            self.additional_scaling = nn.Upsample(mode='bicubic', scale_factor=additional_scaling, align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        x = self.UPNet(x)
        
        if self.scale:
            x = self.additional_scaling(x)
        x = self.sigmoid(x)
        return x
