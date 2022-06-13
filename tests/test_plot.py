import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.datasets import *
from src.plot import *


date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
dataset_planet = SN7Dataset(sat='planet',
                            date_range=date_range,
                            samples=100,
                            window_size=30,
                            random_seed=42, debug=True)


def test_show_tensor_input():
    show((255 * torch.rand(10, 10)).to(int))
    show((255 * torch.rand(4, 10, 10)).to(int))
    show((255 * torch.rand(1, 10, 10)).to(int))
    show((255 * torch.rand(1, 4, 10, 10)).to(int))


def test_show_bool_input():
    show(255 * (torch.rand(10, 10)) > .5)


def test_show_xarray_input():
    x = xr.DataArray((255 * np.random.rand(3, 10, 10)).astype(int), dims=('time', 'y', 'x'))
    show(x)
    show(x, title=[''])
    show(x, title=np.array([1]))
    show(dataset_planet[0]['images'][0])


def test_show_ndarray_input():
    show((255 * np.random.rand(4, 10, 10)).astype(int))


def test_show_4D_input():
    show((255 * np.random.rand(3, 4, 10, 10)).astype(int))
    show((255 * np.random.rand(1, 4, 10, 10)).astype(int))


def test_show_2D_input():
    show((255 * np.random.rand(10, 10)).astype(int), colorbar=True)
    show(np.random.rand(10, 10) > .5, colorbar=True)


def test_show_kwargs():
    show((255 * np.random.rand(1, 3, 10, 10)).astype(int), figsize=(3, 3))
    show((255 * np.random.rand(1, 10, 10)).astype(int), figsize=3)
    show((255 * np.random.rand(10, 10)).astype(int), figsize=3)


def test_show_kwargs():
    show([(255 * np.random.rand(10, 10)).astype(int),
          (255 * np.random.rand(5, 5)).astype(int)])
