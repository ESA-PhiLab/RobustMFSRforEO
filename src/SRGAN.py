# adapted GAN PL training template from https://github.com/homomorfism/SRGAN_PytorchLightning/
# and models from https://github.com/leftthomas/SRGAN/blob/master/model.py
import sys
sys.path.append('..')

from argparse import ArgumentParser, Namespace
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.core import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import math, itertools
import numpy as np
from collections import OrderedDict

from src.datamodules import SN7DataModule
from src.datasources import S2_BANDS
from src.losses import ContentLoss, AdversarialLoss
from src.losses import masked_MSE, cMSE, MSELossFunc
from src.ssim_loss import SSIMLoss, MultiScaleSSIMLoss, ssim
from src.highresnet import ShiftNet
import wandb

class SRResNetModule(pl.LightningModule):
    def __init__(self,  net_args, lr=0.0007, lr_decay=0.97, lr_patience=3, shiftnet=False, loss='MSE'):
        super(SRResNetModule, self).__init__()
        
        self.generator = Generator(in_channels=net_args.in_channels, out_channels=net_args.out_channels, 
                                   upscale_factor=net_args.upscale_factor, additional_scaling=net_args.additional_scaling)
        
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
        return self.generator(lowres_images)

    def prediction_step(self, lowres_dict):
        indices = (lowres_dict['clearances']).argmax(1)
        lowres_images = lowres_dict['images'][torch.arange(len(indices)), indices, ...]

        return self.generator(lowres_images)

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

    
    def test_step(self, batch, batch_idx):
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
        self.log('test_PSNR', - 10 * torch.log10(mse + 1e-8))
        self.log('test_SSIM', ssim(superres, highres_images))
        self.log('test_MSE', mse)
        
        if batch_idx == 0 and self.logger is not None:
            self.logger.experiment.log({"superres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in superres]})
            self.logger.experiment.log({"highres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in highres['images'].squeeze(1)]})
        return {'test_loss': mse}

    def configure_optimizers(self):
        if self.shiftnet:
            optimizer = torch.optim.Adam(list(self.generator.parameters()) + list(self.registration_model.parameters()), self.lr)
        else:
            optimizer = torch.optim.Adam(self.generator.parameters(), self.lr)
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
        

class SRGAN(pl.LightningModule):

    def __init__(self, upscale_factor=2, additional_scaling=1.0469, gen_lr=0.0007, disc_lr=0.0007, adversarial_alpha = 0.001, content_beta = 0.006, 
                 gen_weights_file = None, shiftnet=False):

        super(SRGAN, self).__init__()
        
        self.disc_lr = disc_lr
        self.gen_lr = gen_lr
        
        # Manual LR Scheduler
        self.lr_step_size = 10000
        self.lr_step = 0.1
        
        # Loss and Contributions
        self.adversarial_alpha = adversarial_alpha
        self.content_beta = content_beta
        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss()

        self.generator = Generator(upscale_factor=upscale_factor, additional_scaling=additional_scaling)
        self.shiftnet = shiftnet
        if self.shiftnet:
            self.registration_model = ShiftNet(in_channel=3)
        self.load_gen_weights(gen_weights_file)
        
        self.discriminator = Discriminator()

    def configure_optimizers(self):
        if self.shiftnet:
            optim_generator = torch.optim.Adam(list(self.generator.parameters()) + list(self.registration_model.parameters()), self.gen_lr)
        else:
            optim_generator = torch.optim.Adam(self.generator.parameters(), self.gen_lr)

        optim_discriminator = Adam(self.discriminator.parameters(), lr=self.disc_lr)

        sched_generator = StepLR(optim_generator, step_size=self.lr_step_size, gamma=self.lr_step)
        sched_discriminator = StepLR(optim_discriminator, step_size=self.lr_step_size, gamma=self.lr_step)
        return [optim_generator, optim_discriminator], [sched_generator, sched_discriminator]

    def forward(self, x):
        return self.generator(x)

    def prediction_step(self, lowres_dict):
        indices = (lowres_dict['clearances']).argmax(1)
        lowres_images = lowres_dict['images'][torch.arange(len(indices)), indices, ...]
        return self.generator(lowres_images)

    def training_step(self, batch, batch_idx, optimizer_idx):
        highres = batch['highres']
        lowres = batch['lowres']
        
        indices = (lowres['clearances']).argmax(1)
        lowres_images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
        highres_images = highres['images'].squeeze(1)

        # Generator step
        if optimizer_idx == 0:
            gen_loss = self.generator_loss(lowres_images, highres_images)
            self.log('generator_loss', gen_loss.item())
            return gen_loss

        # Discriminator loss
        else:
            disc_loss = self.discriminator_loss(lowres_images, highres_images)
            self.log('discriminator_loss', disc_loss.item())
            return disc_loss
    
    def validation_step(self, batch, batch_idx):

        highres = batch['highres']
        lowres = batch['lowres']
        
        indices = (lowres['clearances']).argmax(1)
        lowres_images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
        highres_images = highres['images'].squeeze(1)
        batch_size = len(highres_images)
        
        # Generator step
        superres = self.generator(lowres_images)
        if self.shiftnet:
            # register
            # thetas: tensor (B, 2), translation params
            registration_input = torch.cat([highres_images, superres], 1) # concat on channel dimension
            shifts = self.registration_model(registration_input)
            # do registration
            superres = self.registration_model.transform(shifts, superres)
        
        # calculate metrics
        val_score = 0
        for i in range(batch_size):
            val_score += self.shift_loss(np.clip(superres[i].cpu(), 0, 1), highres['images'][i].cpu(), highres['clouds'][i].cpu().squeeze(1))
        val_mse = val_score/batch_size
        self.log('val_MSE', val_score/batch_size)
        
        if batch_idx == 0 and self.logger is not None:
            self.logger.experiment.log({"superres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in superres]})
            self.logger.experiment.log({"highres": [wandb.Image(img.cpu().numpy().transpose(1,2,0)) for img in highres['images'].squeeze(1)]})
        return {'val_loss': val_mse}
    
    def generator_loss(self, lowres_images, highres_images):

        discriminator_fake_outputs = self.discriminator(self.generator(lowres_images))
        adversarial_loss = self.adversarial_loss(discriminator_fake_outputs)
        self.log('adversarial_loss', adversarial_loss.item())

        superres = self.generator(lowres_images)
        if self.shiftnet:
            # register
            # thetas: tensor (B, 2), translation params
            registration_input = torch.cat([highres_images, superres], 1) # concat on channel dimension
            shifts = self.registration_model(registration_input)
            # do registration
            superres = self.registration_model.transform(shifts, superres)

        content_loss = self.content_loss(superres, highres_images)
        self.log('content_loss', content_loss.item())

        g_loss = self.content_beta * content_loss + self.adversarial_alpha * adversarial_loss #+ self.clear_loss(superres, highres_image, clouds_needed)

        return g_loss

    def discriminator_loss(self, lowres_images, highres_images):

        real_loss = torch.log(self.discriminator(highres_images)).mean()
        fake_loss = torch.log(1 - self.discriminator(self.generator(lowres_images))).mean()

        return - (real_loss + fake_loss) / 2
    
    def load_gen_weights(self, gen_weights_file):
        """ Load pre-trained weights of generator. """
        if gen_weights_file is not None:
            state_dict = torch.load(gen_weights_file)['state_dict'] # might need to map_location=self.generator.device
            ## won't be needed for next retrain of SRResNet
            new_state_dict = OrderedDict([(k.replace("net.","generator."), v) for k, v in state_dict.items()])
            ##
            self.load_state_dict(new_state_dict, strict=False)
    
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
        site_MSE = np.array([MSELossFunc(superres, hr, hrm) for hr, hrm in zip(iter_highres, iter_highres_status)])
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

## Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, upscale_factor=2, additional_scaling=1.0469):
        upsample_block_num = int(math.log(upscale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        
        self.scale = False # do not do additional scaling by default
        if additional_scaling is not None:
            self.scale = True
            self.additional_scaling = nn.Upsample(mode='bicubic', scale_factor=additional_scaling, align_corners=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        if self.scale:
            block8 = self.additional_scaling(block8)
        
        # Bound between [0, 1]
        return (torch.tanh(block8) + 1) / 2


## Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        
        return x

def main(args):
    logger = WandbLogger(project="sisr-baselines", entity="fdlsr")
    
    checkpoint_callback = ModelCheckpoint(filepath='/home/muhammed/lightning_saves/',
                                          save_top_k=5,verbose=True, monitor='val_MSE',mode='min',prefix=args.expername)
    
    gen_weights_file = "/home/muhammed/lightning_saves/sisr_SRResNet_SSIM-epoch=5-step=5627.ckpt"
    model = SRGAN(shiftnet=True, gen_weights_file=gen_weights_file)
    
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=logger, check_val_every_n_epoch=2,
                         checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=10, 
                         fast_dev_run=False)
    ## Set up Data Module
    date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
    dm = SN7DataModule(date_range=date_range,s2_bands=S2_BANDS['true_color'], window_size_planet=134, force_size=64, samples=75000, 
                       batch_size=args.batch_size, normalize=True, standardize_sentinel=False)
    dm.setup()
    ## Train
    trainer.fit(model, dm)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expername', type=str, default='sisr', help='Expername')
    parser.add_argument('--gpus', default='0', type=str)
    
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
#    parser.add_argument('--lr', default=0.001, type=float) add gen/disc lr
#
#    parser.add_argument('--loss', type=str, default='MSE', help='')
    
    args = parser.parse_args()
    main(args)
