import sys
sys.path.append('..')

from argparse import ArgumentParser, Namespace
import pandas as pd

from pytorch_lightning.core import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.FSRCNN import FSRCNNModule
from src.SRGAN import SRResNetModule

from src.datamodules import SN7DataModule
from src.datasources import S2_BANDS


def main(args):
    logger = WandbLogger(project="sisr-pretrain", entity="fdlsr")
    logger.log_hyperparams(args)
    
    checkpoint_callback = ModelCheckpoint(dirpath='/home/muhammed/lightning_saves2/',
                                          save_top_k=5,verbose=True, monitor='val_loss',mode='min',prefix=args.expername)
    if args.s2_bands == 3:
        s2_bands = S2_BANDS['true_color']
    elif args.s2_bands == 12:
        s2_bands = range(12)
    else:
        raise NotImplementedError
    net_args = Namespace(in_channels=args.s2_bands, out_channels=3, upscale_factor=2, additional_scaling=1.0469)
    if args.model == 'FSRCNN':
        model = FSRCNNModule(net_args, model='FSRCNN', lr= args.lr, loss=args.loss, shiftnet=args.shiftnet)
    elif args.model == 'SRResNet':
        model = SRResNetModule(net_args, lr= args.lr, loss=args.loss, shiftnet=args.shiftnet)
    elif args.model == 'RDN':
        model = FSRCNNModule(net_args, model='RDN', lr= args.lr, loss=args.loss, shiftnet=args.shiftnet)
    else:
        raise NotImplementedError
    
    
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=logger, check_val_every_n_epoch=2,
                         checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=10, 
                         fast_dev_run=False)
    ## Set up Data Module
    date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
    dm = SN7DataModule(date_range=date_range,s2_bands=s2_bands, window_size_planet=134, force_size=64, samples=100000, 
                       batch_size=args.batch_size, normalize=True, standardize_sentinel=False, num_workers=args.num_workers)
    dm.setup()
    ## Train
    trainer.fit(model, dm)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expername', type=str, default='sisr', help='Expername')
    parser.add_argument('--gpus', default='0', type=str)
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    
    parser.add_argument('--model', type=str, default='FSRCNN', help='which model to train')
    parser.add_argument('--shiftnet', action='store_true')
    parser.add_argument('--s2_bands', default=3, type=int)
    
    parser.add_argument('--loss', type=str, default='MSE', help='')
    
    args = parser.parse_args()
    main(args)

