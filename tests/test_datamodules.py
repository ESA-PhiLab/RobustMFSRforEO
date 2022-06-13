import random

import pandas as pd
import pytest

from src.datasources import *
from src.datamodules import *
from src.datautils import *


dm = SN7DataModule(
    date_range=pd.date_range(start=f"2019-12-30", end=f"2020-01-31"),
    samples=1_000,
    s2_bands=S2_BANDS['true_color'],
    window_size_s2=64,
    random_seed=42,
    batch_size=32,
    labels=True,
    debug=True,
)


def test_SN7DataModule_init():
    with pytest.raises(ValueError):
        WotusDataModule(mode='bad value',
#                         window_size_planet=100, date_range=date_range, normalize=True,
#                         s2_bands=S2_BANDS13['true_color'], samples=1000, debug=True
                       )
#         dm.setup()
#     dm.mode = 'concat'
    with pytest.raises(ValueError):
        SN7DataModule(window_size_planet=134, window_size_s2=64)


def test_SN7DataModule_prepare_setup():
    dm.prepare_data()
    dm.setup()


def test_SN7DataModule_normalize():
    dm.normalize = True
    dm.setup()


def test_SN7DataModule_standardize():
    dm.standardize_sentinel = True
    dm.setup()


def test_SN7DataModule_dataloaders():
    assert dm.train_dataloader().dataset[0]['lowres']['images'][[0]].shape == (1, 3, 64, 64)

    assert len(dm.train_dataloader().dataset) == 800

    assert len(dm.val_dataloader().dataset) == 100

    assert len(dm.test_dataloader().dataset) == 100


def test_SN7DataModule_simulate_epoch():
    B = dm.batch_size
    for i, x in enumerate(iter(dm.train_dataloader())):
        print(f"{i*B}-{(i+1)*B - 1}, {x['highres']['images'].shape}")

    dm.collate_2toN = True
    for i, x in enumerate(iter(dm.val_dataloader())):
        print(f"{i*B}-{(i+1)*B - 1}, {x['lowres']['images'].shape}")


def test_SN7DataModule_train_val_test_split():
    train = set(dm.dataset_train.dataset_sentinel.scenes)
    val = set(dm.dataset_val.dataset_sentinel.scenes)
    test = set(dm.dataset_test.dataset_sentinel.scenes)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    ## Mutual exclusivity among sets.
#     assert len(train) + len(val) + len(test) == len (train | val | test)

    assert len(dm.dataset_train) == 800

    assert len(dm.dataset_val) == 100

    assert len(dm.dataset_test) == 100


def test_SN7DataModule_mode():
    dm.mode = 'planet'
    dm.setup()
    dm.train_dataloader()

    dm.mode = 'sentinel'
    dm.setup()


def test_WotusDataModule():
    date_range = pd.date_range(start=f"2018-08-30", end=f"2019-10-31")
    dm = WotusDataModule(mode='concat', window_size_planet=100, date_range=date_range, normalize=True,
                         s2_bands=S2_BANDS13['true_color'], samples=1000, debug=True)
    dm.prepare_data()
    dm.setup()
    assert dm.train_dataloader().dataset[0]['highres']['images'][[0]].shape == (1, 4, 100, 100)

    dm = WotusDataModule(mode='planet', window_size_planet=100, date_range=date_range, debug=True)
    dm.setup()
    assert dm.train_dataloader().dataset[0]['images'][[0]].shape == (1, 4, 100, 100)

    dm = WotusDataModule(mode='sentinel', window_size_s2=50, date_range=date_range,
                         s2_bands=S2_BANDS13['true_color'], debug=True)
    dm.setup()
    assert dm.train_dataloader().dataset[0]['images'][[0]].shape == (1, 3, 50, 50)

    with pytest.raises(ValueError):
        WotusDataModule(window_size_planet=134, window_size_s2=64)

    with pytest.raises(ValueError):
        WotusDataModule(mode='bad value')
