import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from src.datautils import *



def test_dilate():
    mask = np.zeros((5, 5))
    mask[2, 2] = 1
    assert dilate(mask, iterations=1).sum() == 9


def test_train_val_test_partition():
    x = np.random.rand(100)
    partition = train_val_test_partition(x, split=(.8, .1, .1), seed=1337)
    values = np.array(list(partition.values()))
    assert sum(values == 'train') == 80
    assert sum(values == 'val') == 10
    assert sum(values == 'test') == 10


def test_rand_index_exponential():
    x = np.random.rand(100)
    rand_index_exponential(x, n=10, beta=50, replace=True)
    rand_index_exponential([])


def test_Compose4d():
    x = np.random.rand(10, 5, 5)
    x = [x, x, x]
    x = xr.DataArray(x)
    transform = Compose4d([ToTensor2()])
    assert transform(x).shape == (3, 10, 5, 5)

    x = pd.Series(['a', 'b', 'c'])
    x = [x, x, x]
    with pytest.raises(TypeError):
        transform._transform4d(lambda x: x, x)


def test_ToTensor2():
    x = np.random.rand(5, 5)
    x = [x, x, x]
    x = xr.DataArray(x)
    x2 = ToTensor2(normalize=False)(x)
    assert isinstance(x2, torch.Tensor)
    assert x2.shape == (3, 5, 5)

    x2 = ToTensor2(normalize=True)(x)
    assert isinstance(x2, torch.Tensor)
    assert x2.shape == (3, 5, 5)


def test_Clamp():
    x = torch.rand(5, 5)
    x = Clamp(min=0, max=1)(x)
    assert 0 <= x.min()
    assert x.max() <= 1


def test_Sampler2d():
    n = 100
    p1 = np.random.rand(5, 5)
    p2 = np.zeros((5, 5), dtype=float)
    for p in [p1, p2]:
        sampler = Sampler2d(p=p, n=n)()
        samples = np.array([x for x in sampler])
        assert hasattr(sampler, '__iter__')
        assert len(samples) == n
        assert samples.shape == (n, 2)
        assert (samples < 5).all()


def test_PadCollate():
    x = [torch.ones(1, 1, 5, 5),
         torch.ones(2, 1, 5, 4),
         torch.ones(3, 1, 3, 6),]
    x = PadCollate(dim=(0, 2, 3)).pad_collate(x)
    assert x.shape == (3, 3, 1, 5, 6)

    x = PadCollate(dim=(0, 2, 3), powerOf2=True).pad_collate(x)
    assert x.shape == (3, 4, 1, 5, 6)


def test_scale_window():

    window = scale_window(window=(500, 500), s=(0.5, 0.5))
    assert window == (250, 250)

    window = scale_window(window=(slice(None), slice(None)), s=(1, 1))
    assert window == (slice(None), slice(None))

    window = scale_window(window=(slice(None, 100), slice(None, 50)), s=(1.5, 3))
    assert window == (slice(None, 150), slice(None, 150))

    window = scale_window(window=(slice(None, 10), slice(None, 90)), s=(5.0, 1), force_size=5)
    assert window == (slice(None, 5), slice(None, 5))

    with pytest.raises(TypeError):
        scale_window(window=(100, 100), s='bad value')

    with pytest.raises(TypeError):
        ## Incomplete tuple.
        scale_window(window=(100,), s=(1.0, 1.0))

    with pytest.raises(TypeError):
        ## Incomplete tuple.
        scale_window(window=(100, 100), s=(1.0,))
