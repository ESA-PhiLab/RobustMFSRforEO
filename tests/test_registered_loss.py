from itertools import product
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import cv2
import pandas as pd
import torch
from torch.nn import MSELoss

from src.datamodules import SN7DataModule
from src.datasources import S2_BANDS12
# from src.lanczos import lanczos_shift, lanczos_kernel
from src.registered_loss import RegisteredLoss, ShiftConv2d



def test_ShiftConv2d():
    w = 4
    start, end, step = -(w/2), w/2, 1
    shiftconv2d = ShiftConv2d(start, end, step)
    K_y = shiftconv2d.K_y
    K_x = shiftconv2d.K_x
    K = torch.stack([k_y * k_x for k_y, k_x in product(K_y, K_x)])

    assert K_y.shape == (w+1, 1, 1, 2*(w+1)+1, 1)
    assert K_x.shape == (w+1, 1, 1, 1, 2*(w+1)+1)
    assert K.shape == ((w+1)**2, 1, 1, 2*(w+1)+1, 2*(w+1)+1)


def test_RegisteredLoss():
    d = 20
    img1 = torch.zeros((3, d, d))
    img1[[0], :, d//2] = 1
    img1[[1], d//2, :] = 1
    img2 = torch.zeros(3, d, d)
    img2[0, :, :] = .5 * torch.eye(d).fliplr()
    img2[1, :, :] = .5 * torch.eye(d)
    x = torch.stack([img1, img2])

    w = 4
    start, end, step = -(w/2), w/2, 1
    reg_mse = RegisteredLoss(start, end, step=1, loss_func=MSELoss(reduction='none'), reduction='none')
    x_shifted = reg_mse._shiftconv2d(x)
    assert x_shifted.shape == (2, 25, 3, d, d)

    y = torch.cat([x_shifted[0, [3]], x_shifted[1, [15]]])
    loss_all_shifts = reg_mse._shifted_loss(x, y).detach()
    assert loss_all_shifts.shape == (2, 25)

#     print(MSELoss(reduction='none')(x, y).mean((-3,-2,-1)))
    assert (reg_mse(x, y) == 0).all()


# def test_shifted_loss():
#     y = torch.cat([x_shifted[0, [3]], x_shifted[1, [15]]])
#     loss_all_shifts = reg_mse._shifted_loss(x, y).detach()
#     assert loss_all_shifts.shape == (2, 25)

#     x1 = loss_all_shifts.reshape(2, 5, 5)[0]
#     x2 = loss_all_shifts.reshape(2, 5, 5)[1]

#     show([x1, x2], axis=False, figsize=4, colorbar=True, normalize=False)