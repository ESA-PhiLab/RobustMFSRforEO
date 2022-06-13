from typing import List, Optional, Tuple, Union, Iterable

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from argparse import ArgumentParser, Namespace
import pandas as pd

from pytorch_lightning.core import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.FSRCNN import FSRCNNModule
from src.SRGAN import SRResNetModule
from src.highresnet import HighResNet

from src.datamodules import SN7DataModule
from src.datasources import S2_BANDS
import copy

def models_and_dataset(args, ensemble_size=None, ensemble_checkpoints=None):
    """
    Arguments:
        args (Namespace): sisr, model, batch_size, clouds, shiftnet
        ensemble_size (int): number of ensemble models to return (no weights)
        ensemble_checkpoints (List): list of strings with location of ensemble checkpoints
    Returns:
        models (List): list containing ensembles
        dm (DataModule): PyTorch Lightning DataModule
    """
    if args.sisr:
        net_args = Namespace(in_channels=3, out_channels=3, upscale_factor=2, additional_scaling=1.0469)
        if args.model == 'FSRCNN':
            model = FSRCNNModule(net_args, model='FSRCNN', lr=0.001, loss='MAE', shiftnet=args.shiftnet)
        elif args.model == 'SRResNet':
            model = SRResNetModule(net_args, lr=0.001, loss='MAE', shiftnet=args.shiftnet)
        elif args.model == 'RDN':
            model = FSRCNNModule(net_args, model='RDN', lr=0.001, loss='MAE', shiftnet=args.shiftnet)
        else:
            raise NotImplementedError
        date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
        dm = SN7DataModule(date_range=date_range,s2_bands=S2_BANDS['true_color'], window_size_planet=134, force_size=64, samples=75000, 
                       batch_size=args.batch_size, normalize=True, standardize_sentinel=False)
    else:
        net_args = Namespace(in_channels=3, channel_size=64, out_channels=3, revisits_residual=True, additional_scaling=1.039)
        model = HighResNet(net_args, lr=0.001, lr_decay=0.9, lr_patience=2, clouds=args.clouds, shiftnet=args.shiftnet, loss='MAE')
        ## Set up Data Module
        date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
        dm = SN7DataModule(date_range=date_range,s2_bands=S2_BANDS['true_color'], window_size_planet=134, force_size=64, samples=75000, 
                           batch_size=args.batch_size, collate_2toN=True, normalize=True, standardize_sentinel=False)
        
    dm.setup()
    
    if ensemble_checkpoints is not None:
        models = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for checkpoint_file in ensemble_checkpoints:
            model_instance = copy.deepcopy(model)
            chkpt = torch.load(checkpoint_file, map_location=device)
            model_instance.load_state_dict(chkpt['state_dict'])
            models += [model_instance]
    elif ensemble_size > 0:
        models = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(ensemble_size):
            model_instance = copy.deepcopy(model).to(device)
            models += [model_instance]
    
    return models, dm
            

#### PIQ/Functional/Filters  https://github.com/photosynthesis-team/piq/tree/master/piq
def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)

def _adjust_dimensions(input_tensors: Union[torch.Tensor, Iterable[torch.Tensor]]):
    r"""Expands input tensors dimensions to 4D (N, C, H, W).
    """
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    resized_tensors = []
    for tensor in input_tensors:
        tmp = tensor.clone()
        if tmp.dim() == 2:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() == 3:
            tmp = tmp.unsqueeze(0)
        if tmp.dim() != 4 and tmp.dim() != 5:
            raise ValueError(f'Expected 2, 3, 4 or 5 dimensions (got {tensor.dim()})')
        resized_tensors.append(tmp)

    if len(resized_tensors) == 1:
        return resized_tensors[0]

    return tuple(resized_tensors)

def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: Tensor with shape (N, C, H, W).
        y: Tensor with shape (N, C, H, W).
        kernel: 2D Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu1 = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2 = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (F.conv2d(x * x, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq)
    sigma2_sq = compensation * (F.conv2d(y * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq)
    sigma12 = compensation * (F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    
    return ssim_map, cs_map

def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.

    Args:
        x: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W)
        y: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W)
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        full: Return cs map or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.

    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           DOI: `10.1109/TIP.2003.819861`
    """
    x, y = _adjust_dimensions(input_tensors=(x, y))

    x = x.type(torch.float32)
    y = y.type(torch.float32)
        
    x = x / data_range
    y = y / data_range

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    ssim_map, cs_map = _ssim_per_channel(x=x, y=y, kernel=kernel, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    return ssim_val
