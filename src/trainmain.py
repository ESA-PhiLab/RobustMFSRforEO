import sys
sys.path.append('..')
from argparse import ArgumentParser, Namespace

from pytorch_lightning.core import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.highresnet_lightning import HighResNetLightning
from src.datautils import ConcatPadPTwoCollate
from src.datamodules import SN7DataModule
from src.datasources import S2_BANDS

import pandas as pd

def main(args):
    ## Logging and Saving Options
    logger = WandbLogger(project="multispectral-mfsr", entity="fdlsr")
    logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath='/home/muhammed/lightning_saves2/',
                                          save_top_k=5,verbose=True,monitor=args.loss, mode='min', prefix=args.expername)
    if args.s2_bands == 3:
        s2_bands = S2_BANDS['true_color']
    elif args.s2_bands == 12:
        s2_bands = range(12)
    else:
        raise NotImplementedError
    ## Set up Model
    # additional scaling is a function of the GSD ratios. Upsampling layers at the end to do the additional upsampling
    net_args = Namespace(in_channels=args.s2_bands, num_channels=64, out_channels=3, revisits_residual=True, additional_scale_factor=1.039)
    model = HighResNetLightning(net_args, lr= args.lr, lr_decay=0.9, lr_patience=2, clouds=args.clouds, shiftnet=args.shiftnet, loss=args.loss)
    
    ## Set Up PL Trainer
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=logger, check_val_every_n_epoch=2,
                         checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=10, 
                         fast_dev_run=False)
    ## Set up Data Module
    date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
    dm = SN7DataModule(date_range=date_range, s2_bands=s2_bands, window_size_planet=134, force_size=64, samples=75000, 
                       batch_size=args.batch_size, collate_2toN=True, normalize=True, standardize_sentinel=False)
    dm.setup()
    ## Train
    trainer.fit(model, dm)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expername', type=str, default='higresnet', help='Expername')
    parser.add_argument('--gpus', default='0', type=str)
    
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--clouds', action='store_true')
    parser.add_argument('--shiftnet', action='store_true')
    parser.add_argument('--s2_bands', default=3, type=int)
    
    parser.add_argument('--loss', type=str, default='MSE', help='')
    
    args = parser.parse_args()
    main(args)
