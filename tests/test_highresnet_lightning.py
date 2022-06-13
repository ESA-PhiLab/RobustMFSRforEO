import pytest
import sys

from argparse import ArgumentParser, Namespace
import pandas as pd
import torch
from pytorch_lightning.trainer import Trainer

from src.datasources import S2_BANDS12
from src.datamodules import SN7DataModule
from src.highresnet_lightning import HighResNetLightning


lr = torch.zeros([2, 8, 3, 64, 64])  # (B, T, C, H, W)
lr_clouds = torch.zeros([2, 8, 1, 64, 64])
hr = torch.zeros([2, 1, 3, 134, 134])
hr_clouds = torch.zeros([2, 1, 1, 134, 134])
r = torch.zeros((2, 8))
lr_items = {'images': lr,
            'clouds': lr_clouds,
            'revisits_indicator': r}
hr_items = {'images': hr,
            'clouds': hr_clouds}
items = {'highres': hr_items,
        'lowres': lr_items}

args = Namespace(in_channels=3, num_channels=64, out_channels=3, revisits_residual=True, additional_scale_factor=1.039)

date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
dm = SN7DataModule(date_range=date_range, s2_bands=S2_BANDS12['true_color'], window_size_planet=134, force_size=64, samples=10,
                   batch_size=1, collate_2toN=True, normalize=True, standardize_sentinel=False, labels=True, debug=True)
dm.setup()


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning():
    highresnet = HighResNetLightning(args, lr=0.001, lr_decay=0.9, lr_patience=2, clouds=False, shiftnet=True, loss='MSE')
    sr = highresnet(lr_items)[:, None]  # (B, 1, C, H, W)
    assert sr.shape == hr.shape


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning_clouds():
    highresnet = HighResNetLightning(args, clouds=True, loss='MAE')
    sr = highresnet(lr_items)[:, None]  # (B, 1, C, H, W)
    assert sr.shape == hr.shape


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning_losses():
    highresnet = HighResNetLightning(args, loss='MAE')
    sr = highresnet(lr_items)[:, None]  # (B, 1, C, H, W)
    assert sr.shape == hr.shape

    highresnet = HighResNetLightning(args, loss='SSIM')
    sr = highresnet(lr_items)[:, None]
    assert sr.shape == hr.shape

    highresnet = HighResNetLightning(args, loss='MSSSIM')
    sr = highresnet(lr_items)[:, None]
    assert sr.shape == hr.shape

    highresnet = HighResNetLightning(args, loss='masked_MSE')
    sr = highresnet(lr_items)[:, None]
    assert sr.shape == hr.shape

    highresnet = HighResNetLightning(args, loss='cMSE')
    sr = highresnet(lr_items)[:, None]
    assert sr.shape == hr.shape

    with pytest.raises(ValueError):
        highresnet = HighResNetLightning(args, loss='bad value')


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning_prediction_step():
    highresnet = HighResNetLightning(args)
    sr = highresnet.prediction_step(lr_items)[:, None]  # (B, 1, C, H, W)
    assert sr.shape == hr.shape


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning_training_step():
    highresnet = HighResNetLightning(args)
    loss = highresnet.training_step(batch=items, batch_idx=0)


def test_highresnet_lightning_shiftnet():
    highresnet = HighResNetLightning(args, shiftnet=True)


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_highresnet_lightning_fast_dev_run():
    highresnet = HighResNetLightning(args)
    trainer = Trainer(fast_dev_run=True, gpus=None)
    trainer.fit(highresnet, dm)
    trainer.test(ckpt_path=None)

    ## With ShiftNet
    highresnet = HighResNetLightning(args, shiftnet=True)#, loss='masked_MSE')
    trainer.fit(highresnet, dm)

#     ## With ShiftNet and masked_MSE loss
#     highresnet = HighResNetLightning(args, shiftnet=True, loss='cMSE')
#     trainer.fit(highresnet, dm)


# @pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
# def test_highresnet_lightning_shift_MSE():
#     for loss in ['MSE']:#, 'masked_MSE', 'cMSE']:
#         highresnet = HighResNetLightning(args, shiftnet=True, loss=loss)
#         sr = highresnet(lr_items)
#         highresnet.shift_MSE(sr[0], hr[0][0], hr_clouds[0][0], border_w=3)


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_highresnet_lightning_clear_loss():
    highresnet = HighResNetLightning(args, shiftnet=True, loss='SSIM')
    sr = highresnet(lr_items)
    highresnet.clear_loss(sr, hr[:, 0], hr_clouds[:, 0])

#     highresnet = HighResNetLightning(args, shiftnet=True, loss='masked_MSE')
#     sr = highresnet(lr_items)
#     highresnet.clear_loss(sr, hr[:, 0], hr_clouds[:, 0])

#     highresnet = HighResNetLightning(args, shiftnet=True, loss='cMSE')
#     sr = highresnet(lr_items)
#     highresnet.clear_loss(sr, hr[:, 0], hr_clouds[:, 0])


# @pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
# def test_highresnet_lightning_MSEtoPSNR():
#     highresnet = HighResNetLightning(args, shiftnet=True, loss='SSIM')
#     sr = highresnet(lr_items)
#     loss = highresnet.clear_loss(sr, hr[:, 0], hr_clouds[:, 0])
#     highresnet.MSEtoPSNR(loss)
