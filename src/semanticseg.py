import sys
sys.path.append('..')
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import MetricCollection, IoU, Accuracy
from src.datamodules import SN7DataModule
from src.hrnet_seg import HighResolutionNet
from src.unet import UNet
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from src.datasources import S2_BANDS
from src.registered_loss import RegisteredLoss

from src.highresnet_lightning import HighResNetLightning
from src.FSRCNN import FSRCNNModule
from src.SRGAN import SRResNetModule

import wandb
class_labels = {
    0: "not building",
    1: "building"
}


class SegModel(pl.LightningModule):
    def __init__(self, model='HRNet', loss='BCE', lr=0.001, dataset='planet', registered_loss=False, 
                 model_ckpt=None, superres_model=None, **kwargs):
        super().__init__(**kwargs)
        
        self.lr = lr
        self.lr_decay=0.9
        self.lr_patience=2
        
        self.dataset = dataset
        if self.dataset == 'sentinel' or self.dataset == 'sentinel_multi':
            self.upsample = nn.Upsample(mode='bicubic', scale_factor=2.094, align_corners=False)
        
        if model =='HRNet':
            if self.dataset == 'sentinel_multi':
                self.net = HighResolutionNet(input_channels=12)
            else:
                self.net = HighResolutionNet(input_channels=3)
        elif model =='UNet':
            self.net = UNet()
        else:
            raise NotImplementedError
        
        if superres_model is not None:
            self.load_superres(superres_model)

            
        if loss == 'BCE' and registered_loss == False:
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif loss == 'BCE' and registered_loss == True:
            self.loss = RegisteredLoss(start=-3, end=3, step=0.5, loss_func=torch.nn.BCEWithLogitsLoss(reduction='none'))
        elif loss == 'softIOU':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.val_metrics = MetricCollection([Accuracy(),
                                             IoU(num_classes=2)
                                            ])
            
    def load_superres(self, superres_model):
        if superres_model == 'MFSR':
            ## NOTE MFSR MODEL REQUIRES Collate2N Padding in DataModule
            net_args = Namespace(in_channels=3, num_channels=64, out_channels=3, revisits_residual=True, additional_scale_factor=1.039)
            self.superres_model = HighResNetLightning(net_args, lr=0.001, lr_decay=0.9, lr_patience=2, clouds=False, shiftnet=True, loss='MAE')
#             model_ckpt = '/home/muhammed/gs_models/hrnet_SSIM_en0-epoch=24-step=46874.ckpt'
            model_ckpt = '/home/muhammed/gs_models/new_data/MFSR_SSIM_newdata-lightning_saves2-v3.ckpt'
        else:
            net_args = Namespace(in_channels=3, out_channels=3, upscale_factor=2, additional_scaling=1.0469)
            if superres_model == 'FSRCNN':
                model_ckpt = '/home/muhammed/gs_models/sisr_FSRCNN_SSIM-epoch=7-step=7503.ckpt'
                self.superres_model = FSRCNNModule(net_args, model='FSRCNN', lr=0.001, loss='MAE', shiftnet=True)
            elif superres_model == 'SRResNet':
#                 model_ckpt = '/home/muhammed/sisr_SRRES_SSIM-lightning_saves2-v0.ckpt'
                model_ckpt = '/home/muhammed/gs_models/new_data/sisr_SRRES_SSIM-lightning_saves2-v3.ckpt'
                self.superres_model = SRResNetModule(net_args, lr=0.001, loss='MAE', shiftnet=True)
            elif superres_model == 'RDN':
                raise NotImplementedError
                self.superres_model = FSRCNNModule(net_args, model='RDN', lr=0.001, loss='MAE', shiftnet=True)
            else:
                raise NotImplementedError

        device = self.device
        chkpt = torch.load(model_ckpt, map_location=device)
        self.superres_model.load_state_dict(chkpt['state_dict'], strict=True)
        for param in self.superres_model.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if self.dataset == 'planet':
            images = batch['highres']['images'].squeeze(1)
        elif self.dataset == 'sentinel':
            indices = (batch['lowres']['clearances']).argmax(1)
            images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
            images = self.upsample(images)
        elif self.dataset == 'superres':
            with torch.no_grad():
                superres = self.superres_model.prediction_step(batch['lowres'])
                registration_input = torch.cat([batch['highres']['images'].squeeze(1), superres], 1) # concat on channel dimension
                shifts = self.superres_model.registration_model(registration_input)
                images = self.superres_model.registration_model.transform(shifts, superres)
                images = torch.clamp(images, min=0.0, max=1.0)
        elif self.dataset == 'sentinel_multi':
            B, T, C, H, W = batch['lowres']['images'].shape
            device = batch['lowres']['images'].device
            images = torch.zeros([B, 4, C, H, W], device=device)
            indices = (batch['lowres']['clearances']).argsort(dim=1, descending=True)[:,:4]
            for i in range(B):
                images[i] = batch['lowres']['images'][i, indices[i]]
            images = self.upsample(images.view(B, 4*C, H, W))
            
        out = self.forward(images)
        labels = batch['highres']['labels'].squeeze(1)

        loss = self.loss(out, labels)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.dataset == 'planet':
            images = batch['highres']['images'].squeeze(1)
        elif self.dataset == 'sentinel':
            indices = (batch['lowres']['clearances']).argmax(1)
            images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
            images = self.upsample(images)
        elif self.dataset == 'sentinel_multi':
            B, T, C, H, W = batch['lowres']['images'].shape
            device = batch['lowres']['images'].device
            images = torch.zeros([B, 4, C, H, W], device=device)
            indices = (batch['lowres']['clearances']).argsort(dim=1, descending=True)[:,:4]
            for i in range(B):
                images[i] = batch['lowres']['images'][i, indices[i]]
            images = self.upsample(images.view(B, 4*C, H, W))
        elif self.dataset == 'superres':
            with torch.no_grad():
                superres = self.superres_model.prediction_step(batch['lowres'])
                registration_input = torch.cat([batch['highres']['images'].squeeze(1), superres], 1) # concat on channel dimension
                shifts = self.superres_model.registration_model(registration_input)
                images = self.superres_model.registration_model.transform(shifts, superres)
                images = torch.clamp(images, min=0.0, max=1.0)

        out = self.forward(images)
        labels = batch['highres']['labels'].squeeze(1)
        loss = self.loss(out, labels)
        self.val_metrics((torch.sigmoid(out)>0.5).int(), labels.int())
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)
        
        if batch_idx == 0 and self.logger is not None:
            wandb_images = [(img.cpu().numpy().transpose(1,2,0)[:,:,:3]) for img in images]
            wandb_labels = [(img.squeeze().cpu().numpy()) for img in labels]
            wandb_preds = [((torch.sigmoid(img.squeeze())>0.5).int().cpu().numpy()) for img in out]
            wandb_mask = []
            for i in range(len(wandb_images)):
                wandb_mask += [wb_mask(wandb_images[i], wandb_preds[i], wandb_labels[i])]
            self.logger.experiment.log({"predictions" : wandb_mask})
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_decay, verbose=True, patience=self.lr_patience)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def prediction(self, batch):
        if self.dataset == 'planet':
            images = batch['highres']['images'].squeeze(1)
        elif self.dataset == 'sentinel':
            indices = (batch['lowres']['clearances']).argmax(1)
            images = batch['lowres']['images'][torch.arange(len(indices)),indices,...]
            images = self.upsample(images)
        elif self.dataset == 'superres':
            with torch.no_grad():
                superres = self.superres_model.prediction_step(batch['lowres'])
                registration_input = torch.cat([batch['highres']['images'].squeeze(1), superres], 1) # concat on channel dimension
                shifts = self.superres_model.registration_model(registration_input)
                images = self.superres_model.registration_model.transform(shifts, superres)
                images = torch.clamp(images, min=0.0, max=1.0)
        
        return self.forward(images)

def main(args):
    # Logging
    logger = WandbLogger(project="segmentation_model", entity="fdlsr")
    checkpoint_callback = ModelCheckpoint(dirpath='/home/muhammed/lightning_saves2/',
                                          save_top_k=5,verbose=True,monitor='val_loss',mode='min',prefix=args.expername)
        
    # Model
    model = SegModel(model=args.model, loss=args.loss, lr=args.lr, dataset=args.dataset, registered_loss=False, 
                     model_ckpt=None, superres_model=args.superres)
    
    # Trainer
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=logger, check_val_every_n_epoch=2,
                      checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=10,
                      fast_dev_run=False)
    # Datamodule
    date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
    dm = SN7DataModule(date_range=date_range,s2_bands=S2_BANDS['true_color'], window_size_planet=134, force_size=64, samples=75000,
                       batch_size=args.batch_size, normalize=True, standardize_sentinel=False, labels=True, collate_2toN=True)
    dm.setup()
    
    ## Train
    trainer.fit(model, dm)

def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : class_labels},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : class_labels}})

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expername', type=str, default='semanticsegmentation', help='Expername')
    parser.add_argument('--gpus', default='0', type=str)
    
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    
    parser.add_argument('--dataset', type=str, default='planet', help='planet/sentinel/superres')
    parser.add_argument('--superres', type=str, help='MFSR/FSRCNN/SRResNet/RDN')
    
    parser.add_argument('--model', type=str, default='HRNet', help='')
    parser.add_argument('--loss', type=str, default='BCE', help='')
    parser.add_argument('--registered', action='store_true')
    
    args = parser.parse_args()
    main(args)
