import random
from typing import Callable, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize

from src.datasets import *
from src.datautils import *

DEFAULT_SEED = 1337



class SatelliteDataModule(VisionDataModule):

    name = "satellite"
    dataset_cls = SatelliteDataset
    concat_dataset_cls = ConcatSatelliteDataset

    def __init__(
        self,
        mode : str = 'concat',
        window_size_planet : Optional[int] = None,
        window_size_s2 : Optional[int] = None,
        force_size: Optional[int] = None,
        date_range : Optional[pd.DatetimeIndex] = None,
        s2_bands : Optional[List[int]] = None,
        labels : bool = False,
        samples : int = 1000,
        split_scene_mode : str = 'within',
        only_whole_scenes : bool = False,
        random_seed : int = DEFAULT_SEED,
        collate_2toN: bool = False,
        standardize_sentinel: bool = False,
        debug : bool = False,
        HSV : bool = False,
        *args : Any,
        **kwargs : Any,
    ) -> None:
        """
        Args:
            mode (str, default='concat'): One of ['concat'|'planet'|'sentinel'] mode of operations.
                'concat' serves items from a ConcatDataset object.
                'planet' and 'sentinel' serve items from SpaceNet7Dataset objects.
            window_size_planet (int, optional): If mode is 'concat'|'planet', this is the size of the
                square cropping window on Planet images, and implies same-span crops from Sentinel.
            window_size_s2 (int, optional): If mode is 'concat'|'sentinel', this is the size of the
                square cropping window on Sentinel images, and implies same-span crops from Planet.
                At most one of `window_size_planet` and `window_size_s2` can be defined.
            force_size (int, optional): Size of the low res cropping window.
            date_range (pandas.DatetimeIndex, optional): Date range of scenes.
            s2_bands (list of int, optional): Sentinel-2 band indices to get.
            labels (bool, default=False): Read PlanetScope labels (assuming they exist).
                These are read from paths sourced from `self.dataset_planet.df["label_mask_path"]`.
            samples (int, default=1000): Number of samples / random patches to crop out of all scenes.
            split_scene_mode (str): One of ['within'|'across'].
                If 'within', each scene is partitioned to 80% train, 10% val, 10% test areas.
                If 'across', the set of scenes is partitioned to approx 80% train, 10% val, 10% test sets.
            only_whole_scenes (bool, default=False): Serves the full images fo all scenes.
                No random patch sampling. Overrides `samples` and `window_size`.
            random_seed (int, default=DEFAULT_SEED): Random seed for reproducible sampling of patches.
            collate_2toN (bool, default=False): Revisits are padded to the ceiling 2^n, or to the ceiling number if False.
            normalize (bool, default=False): Apply the image Normalize transform.
            standardize_sentinel (bool, default=False): Standardize S2 data.
            num_workers (int, default=16): How many workers to use for loading data.
            batch_size (int, default=32): Number of samples per minibatch.
            shuffle (bool, default=False): If true shuffles the train data every epoch.
            pin_memory (bool, default=False): If true, the data loader will copy Tensors into CUDA pinned memory before returning them.
            drop_last (bool, default=False): If true drops the last incomplete batch.
            debug (bool, default=False): Used for debugging sessions. Limits the dataset to a few scenes.
        """

        super().__init__(*args, **kwargs)

        if mode not in ['concat', 'sentinel', 'planet']:
            raise ValueError(f"mode must be one of ['concat'|'sentinel'|'planet']. Got '{mode}'.")
        self.mode = mode

        if window_size_planet and window_size_s2:
            raise(ValueError("At most one of `window_size_planet` and `window_size_s2` can be defined."))
        self.window_size_planet = window_size_planet

        self.window_size_s2 = window_size_s2
        self.force_size = force_size
        self.date_range = date_range
        self.s2_bands = s2_bands or list(range(13))
        self.labels = labels
        self.samples = samples
        self.split_scene_mode = split_scene_mode
        self.only_whole_scenes = only_whole_scenes
        self.seed = random_seed
        self.collate_2toN = collate_2toN
        self.standardize_sentinel = standardize_sentinel
        self.debug = debug

        self.dataset_cls = self.__class__.dataset_cls
        self.concat_dataset_cls = self.__class__.concat_dataset_cls


    def prepare_data(self, *args : Any, **kwargs : Any) -> None:
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test datasets.
        """

        n_train, n_val, n_test = [int(self.samples * p) for p in (0.8, 0.1, 0.1)]

        train_transforms = self.train_transforms or self.default_transforms()
        val_transforms = self.val_transforms or self.default_transforms()
        test_transforms = self.test_transforms or self.default_transforms()

        dataset_planet_kwargs = dict(sat='planet', window_size=self.window_size_planet, labels=self.labels, date_range=self.date_range,
                                     only_whole_scenes=self.only_whole_scenes, random_seed=self.seed,
                                     split_scene_mode=self.split_scene_mode, debug=self.debug)

        dataset_sentinel_kwargs = dict(sat='sentinel', window_size=self.window_size_s2, date_range=self.date_range, bands=self.s2_bands,
                                       only_whole_scenes=self.only_whole_scenes, random_seed=self.seed,
                                       split_scene_mode=self.split_scene_mode, debug=self.debug)

        concatdataset_kwargs = dict(window_size_planet=self.window_size_planet, window_size_s2=self.window_size_s2,
                                    force_size=self.force_size, random_seed=self.seed)

        P_train, P_val, P_test, P_all = None, None, None, None

        D_train = {'planet': None, 'sentinel': None}
        D_val = {'planet': None, 'sentinel': None}
        D_test = {'planet': None, 'sentinel': None}
        D_all = {'planet': None, 'sentinel': None}
        dataset_kwargs = {'planet': dataset_planet_kwargs, 'sentinel': dataset_sentinel_kwargs}

        for sat in ['planet', 'sentinel']:
            if self.mode in ['concat', sat]:
                D_train[sat] = self.dataset_cls(train_val_test='train', transform=train_transforms,
                                                samples=n_train, **dataset_kwargs[sat])
                D_val[sat] = self.dataset_cls(train_val_test='val', transform=val_transforms,
                                              samples=n_val, **dataset_kwargs[sat])
                if stage == "test" or stage is None:
                    D_test[sat] = self.dataset_cls(train_val_test='test', transform=test_transforms,
                                                   samples=n_test, **dataset_kwargs[sat])
                D_all[sat] = self.dataset_cls(train_val_test=None, transform=test_transforms,
                                              samples=self.samples, **dataset_kwargs[sat])
                if self.mode == sat:
                    self.dataset_train, self.dataset_val = D_train[sat], D_val[sat]
                    self.dataset_test, self.dataset_all = D_test[sat], D_all[sat]

        if self.mode == 'concat':
            self.dataset_train = self.concat_dataset_cls((D_train['planet'], D_train['sentinel']),
                                                         transform=train_transforms, **concatdataset_kwargs)
            self.dataset_val = self.concat_dataset_cls((D_val['planet'], D_val['sentinel']),
                                                       transform=val_transforms, **concatdataset_kwargs)
            if stage == "test" or stage is None:
                self.dataset_test = self.concat_dataset_cls((D_test['planet'], D_test['sentinel']),
                                                            transform=test_transforms, **concatdataset_kwargs)
            self.dataset_all = self.concat_dataset_cls((D_all['planet'], D_all['sentinel']),
                                                       transform=test_transforms, random_seed=self.seed)


    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        if self.mode == 'concat':
            collate_fn = ConcatPadPTwoCollate() if self.collate_2toN else ConcatPadCollate()
        else:
            collate_fn = PadCollate()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, drop_last=self.drop_last,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)


    def default_transforms(self) -> Callable:
        """
        Default transform applied on a ConcatSatelliteDataset item.
        """

        self.transform_planet = Compose4d([ToTensor2()])
        self.transform_sentinel = Compose4d([ToTensor2()])
        self.transform_cloud = Compose4d([ToTensor2(normalize=False)])

        def _append_normalizer(T, mean, std, standardize=False):
            if mean is not None and std is not None:
                if standardize:
                    normalizer = Normalize(mean=mean, std=std)
                else:
                    max95 = list(np.array(mean) + 2*np.array(std))
                    list0 = [0] * len(max95)
                    normalizer = Normalize(mean=list0, std=max95)
                T.transforms.append(normalizer)
                T.transforms.append(Clamp(min=0, max=1))
            return T

        def concat_transform(item : Dict) -> Dict:
            self.transform_planet = Compose4d([ToTensor2()])
            self.transform_sentinel = Compose4d([ToTensor2()])
            self.transform_cloud = Compose4d([ToTensor2(normalize=False)])
            T_p, T_s, T_cloud = self.transform_planet, self.transform_sentinel, self.transform_cloud
            if self.normalize:
                mean_s = self.dataset_train.dataset_sentinel.mean
                std_s = self.dataset_train.dataset_sentinel.std
                T_s = _append_normalizer(T_s, mean=mean_s, std=std_s, standardize=self.standardize_sentinel)
            item['highres']['images'] = T_p(item['highres']['images'])
            item['highres']['clouds'] = T_cloud(item['highres']['clouds'])
            item['lowres']['images'] = T_s(item['lowres']['images'])
            item['lowres']['clouds'] = T_cloud(item['lowres']['clouds'])
            return item

        if self.mode == 'concat':
            T = concat_transform
        elif self.mode in ['planet', 'sentinel']:
            sat = self.mode
            if sat == 'planet':
                T = self.transform_planet
            elif sat == 'sentinel':
                T = self.transform_sentinel
            if self.normalize:
                mean, std = self.dataset_cls.mean[sat], self.dataset_cls.std[sat]
                T = _append_normalizer(T, mean=mean, std=std,
                                       standardize=(sat == 'sentinel') and self.standardize_sentinel)

        return T



class SN7DataModule(SatelliteDataModule):
    """
    Specs:
        - PlanetScope: (3 bands, H1, W1).
        - Sentinel-2: (12 bands + 1 cloud, H2, W2). See `datasources.S2_BANDS12`.

    Example::
        from src.datamodules import SN7DataModule
        dm = SN7DataModule(date_range=date_range, s2_bands=S2_BANDS12['true_color'],
                           window_size_planet=64, samples=1_000, batch_size=1)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "sn7"
    dataset_cls = SN7Dataset

    def __init__(self, *args : Any, **kwargs : Any) -> None:
        super().__init__(*args, **kwargs)



class WotusDataModule(SatelliteDataModule):
    """
    Specs:
        - PlanetScope: (4 bands, H1, W1).
        - Sentinel-2: (13 bands + 1 cloud, H2, W2). See `datasources.S2_BANDS13`.

    Example::
        from src.datamodules import WotusDataModule
        dm = WotusDataModule(date_range=date_range, s2_bands=S2_BANDS13['true_color'],
                             window_size_planet=64, samples=1_000, batch_size=1)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "wotus"
    dataset_cls = WotusDataset

    def __init__(self, *args : Any, **kwargs : Any) -> None:
        super().__init__(*args, **kwargs)
