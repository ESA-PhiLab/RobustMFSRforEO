from collections import OrderedDict
from typing import Any, List, Union, Optional, Tuple, Callable, Dict, Iterable
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import xarray as xr

from src.datautils import *
from src.datasources import spacenet7_index, wotus_index, MNT_SPACENET7, MNT_WOTUS, TRAIN_VAL_TEST_SPLIT_SN7

Array = Union[xr.DataArray, np.ndarray, torch.Tensor]
DEFAULT_SEED = 1337



class Item(dict):
    def __str__(self) -> str:
        return (
            f"scene {self['scene']}\n"
            f"revisits {self['revisits']}\n"
            f"dates {self['dates'].astype(str).tolist()}\n"
            f"images {self['images'].shape}\n"
            f"clouds {tuple(self['clouds'].shape)}\n"
            f"labels {tuple(self['labels'].shape) if 'labels' in self else None}\n"
            f'clearances {[round(float(c), 2) for c in self["clearances"]]}\n'
            f"band means {[round(float(m), 2) for m in np.array(self['images']).mean(axis=(0, 2, 3))]}"
        )



class SatelliteDataset(Dataset):
    """
    SpaceNet7 abstract dataset class.
    """

    mean : Dict[str, List[float]] = {}
    std : Dict[str, List[float]] = {}
    MNT : str
    index : Callable
    default_bands : Dict[str, list] = {}
    xarray_cache : Dict[str, Any] = {}

    def __init__(
        self,
        #--- back-compatibility w Dataset ---
        data_dir : Optional[str] = None,
        download : Optional[str] = None,
        train : Optional[bool] = None,
        #------------------------------------
        df : Optional[pd.DataFrame] = None,
        sat : Optional[str] = None,
        bands : Optional[List[int]] = None,
        labels : bool = False,
        split_scene_mode : str = 'within',
        train_val_test : Optional[str] = None,
        date_range: Optional[pd.DatetimeIndex] = None,
        window_size : Optional[int] = None,
        samples : int = 1000,
        sample_by_clearance : bool = False,
        only_whole_scenes : bool = False,
        transform : Optional[Callable] = None,
        random_seed : int = DEFAULT_SEED,
        debug : bool = False,
    ) -> None:
        """
        Args:
            df (pandas.DataFrame): Dataframe of image paths.
            sat (str): Name of the satellite source.
            bands (list of int): Band indices to get.
            labels (bool, default=False): Read labels (assuming they exist).
                These are read from paths sourced from `df["label_mask_path"]`.
            split_scene_mode (str): One of ['within'|'across'].
                If 'within', each scene is partitioned to 80% train, 10% val, 10% test areas.
                If 'across', the set of scenes is partitioned to approx 80% train, 10% val, 10% test sets.
            train_val_test (str, optional): One of ['train'|'val'|'test'] or None.
            date_range (pandas.DatetimeIndex, optional): Date range of scenes.
                Overrides `year` and `month`.
            window_size (int, (int, int), optional): Size of the cropping window.
            samples (int, default=1000): Number of samples / random patches
                to crop out of the pool of scenes.
            sample_by_clearance (bool, default=False): Samples patches conditional
                on clearance.
            only_whole_scenes (bool, default=False): Serves the full image of a scenes.
                No random patch sampling. Overrides `samples` and `window_size`.
            transform (callable, optional): Transform to be applied on an item.
            random_seed (int, default={datasets.DEFAULT_SEED}): For the reproducible sampling of patches.
            debug (bool, default=False): Used for debugging sessions. Limits the
                dataset to a few scenes.
        """

        self.data_dir = data_dir

        self.sat = sat
        self.bands = bands
        if split_scene_mode not in ['within', 'across']:
            raise ValueError(f'split_scene_mode must be one of ["within"|"across"]. Got {split_scene_mode}.') 
        self.split_scene_mode = split_scene_mode
        self.train_val_test = train_val_test
        self.labels = labels

        self.transform = transform
        self.samples = samples
        self.only_whole_scenes = only_whole_scenes
        self.scene_clearances = OrderedDict()  # memory of order of entry
        self.xarrays = {}
        self.xarrays_cloud = {}
        self.xarrays_labels = {}
        self.scenes = np.array([])
        self.random_seed = random_seed
        self.df = df  # Dataset image index
        self.sample_by_clearance = sample_by_clearance
        self.xarrays_clearance_prob = {}
        self.scene_patch_generators = {}
        self.window_size = window_size
        self.date_range = date_range
        self.debug = debug

        if type(self) is not SatelliteDataset:
            self._init_scenes()
            self._init_xarrays()
            self._init_clearances()
            self._init_scene_patch_generators()
            self._init_patches()


    def cloud_masks(self, *args : Any, **kwargs : Any) -> None:
        raise NotImplementedError("cloud_masks() is undefined for this abstract class.")


    def clearance_masks(self, *args : Any, **kwargs : Any) -> None:
        raise NotImplementedError("clearance_masks() is undefined for this abstract class.")


    @property
    def window_size(self) -> None:
        """ Getter for `self.window_size`. """
        return self._window_size


    @property
    def random_seed(self) -> None:
        """ Getter for `self.random_seed`. """
        return self._random_seed


    @window_size.setter
    def window_size(self, ws : Optional[int]) -> None:
        """
        Setter for `self.window_size`.
        Also resets `self.patches` using the new window size.
        """
        if not isinstance(ws, int) and ws is not None:
            raise TypeError(f'window_size expected type: int or None. Got {ws}.')
        self._window_size = ws
        if len(self.scenes) > 0:
            self._init_patches()  # Random patches with the new window size.


    @random_seed.setter
    def random_seed(self, seed : int) -> None:
        self._random_seed = seed
        self._rng = np.random.RandomState(seed)
        self._random_state = self._rng.get_state()


    def _init_scenes(self) -> None:
        """
        Init `scenes` property with a pd.Series of scene IDs.
        """

        ## Restrict to train / val / test scenes.
        if self.train_val_test in ['train', 'val', 'test']:
            if self.split_scene_mode == 'within':
                pass
            if self.split_scene_mode == 'across':
                self.df = self.df.query(f'split=="{self.train_val_test}"')
        elif self.train_val_test is not None:
            raise ValueError(f'train_val_test must be one of ["train"|"val"|"test"] or None. Got {self.train_val_test}.')

        ## Restrict to scenes of this satellite.
        if self.sat:
            self.df = self.df.query(f'sat=="{self.sat}"')

        ## Restrict to scenes of this date range.
        if isinstance(self.date_range, pd.DatetimeIndex):
            ix = self.df.index.get_level_values('datetime').isin(self.date_range)
            self.df = self.df.loc[ix]
        elif self.date_range is not None:
            raise TypeError(f'date_range must be a pandas.DatetimeIndex. Got {type(self.date_range)}.')

        ## Series of unique scenes, with index 0,1,2...
        self.scenes = (self.df.index
                       .get_level_values('scene')  # "scene" level of the multi-index
                       .unique())
        self.scenes = self.scenes.to_series(index=range(len(self.scenes)))

        ## Limit scenes for debugging sessions.
        if self.debug:
            self.scenes = self.scenes.iloc[:1]
            ix = self.df.index.get_level_values('scene').isin(self.scenes)
            self.df = self.df.loc[ix]
            for s in self.scenes:
                ix = self.df.query(f'scene == "{s}"').index
                #ix = ix[:max(1, int(len(ix)) * .1)]
                self.df.drop(index=ix[1:], inplace=True)

        ## Issue warning when no scenes are found.
        if len(self.scenes) == 0:
            warnings.warn('No scenes were found at Dataset creation.', UserWarning)


    def _clearance_probability(self, clearance_masks):
        p = (~dilate(~clearance_masks, iterations=8)).sum(axis=0)
        if p.sum() == 0:
            p[:] = 1  # Uniform prob on area excluding strips
        return p / p.sum()


    def _get_xarray_scene(self, scene : str) -> Tuple[Array, Array]:
        """ Returns the xarray of a single scene. """

        ## Read images for this scene and satellite.
        df = self.df.loc[(self.sat, scene)]
        paths, dates = df['path'], xr.Variable('time', df.index)
        X = [xr.open_rasterio(p) for p in paths]
        L = []
        if self.labels:
            L = [xr.open_rasterio(p)[0] for p in df['label_mask_path']]

        ## Cloud masks. xarrays at this point.
        C = self.cloud_masks(scene=scene, images=X)

        ## Bands.
        X = [x[self.bands] for x in X]

        ## TODO: interp_like must be done on each channel individually.
        for D in (X, C, L):
            assert len(set([x.shape for x in D])) <= 1, "Non-matching shapes."

        X = xr.concat(X, dim=dates)
        C = xr.concat(C, dim=dates).round().astype(np.uint8)

        ## Clearance probability.
        P = self._clearance_probability(self.clearance_masks(C).values)
        P = xr.DataArray(P, coords=[C.y, C.x], dims=["y", "x"])

        if self.labels:
            L = xr.concat(L, dim=dates).round().astype(np.uint8)
            return X, C, P, L
        else:
            return X, C, P


    def _get_cached_xarray(
        self,
        sat: str,
        scene: str,
        date_start_stop: Tuple[str, str],
        bands: Tuple[int],
        labels: bool,
    ) -> Tuple[Array]:
        """ Gets xarray from cache if it exists, otherwise it is cached. """
        cache = self.__class__.xarray_cache
        hash_ = (sat, scene, date_start_stop, bands, labels)  # Hashable
        if hash_ not in cache:
            cache[hash_] = self._get_xarray_scene(scene)
        return cache[hash_]


    def _init_xarrays(self) -> None:
        """
        Initializes `self.xarrays` for the PlanetScope and Sentinel Dataset classes.
        """

        def _init_scene(scene: str) -> None:
            if self.date_range is not None:
                dates = self.date_range.date
                date_start_stop = (str(dates.min()), str(dates.max()))
            else:
                date_start_stop = ('', '')
            hash_ = (self.sat, scene, date_start_stop, tuple(self.bands), self.labels)  # Hashable
            return self._get_cached_xarray(*hash_)

        ## TODO: merge all xarrays in one Dict
        if self.sat:
            self.xarrays, self.xarrays_cloud, self.xarrays_clearance_prob, self.xarrays_labels = {}, {}, {}, {}  # Reset
            bar = tqdm(self.scenes, leave=False, desc=f'Loading {self.sat} scenes xarrays...')
            for sc in bar:
                if self.labels:
                    self.xarrays[sc], self.xarrays_cloud[sc], self.xarrays_clearance_prob[sc], self.xarrays_labels[sc] = _init_scene(sc)
                else:
                    self.xarrays[sc], self.xarrays_cloud[sc], self.xarrays_clearance_prob[sc] = _init_scene(sc)


    def _init_clearances(self) -> None:
        """ Cache the clearance scores for the current pool of scenes. """
        if self.sat:
            self.scene_clearances = {}  # Reset
            for scene in self.scenes:
                clouds = self.xarrays_cloud[scene]
                clear_masks = self.clearance_masks(clouds)  # Same shape bool mask.
                self.scene_clearances[scene] = clear_masks.values.astype(float).mean()


    def _init_scene_patch_generators(self) -> None:
        if self.sample_by_clearance:
            self._rng.set_state(self._random_state)  # Reset the RNG state
            ws = self.window_size
            for s in self.scenes:
                p = self.xarrays_clearance_prob[s].values
                if ws:
                    p[-ws:, :] = 0  # Exclude right and bottom strips
                    p[:, -ws:] = 0
                sampler = Sampler2d(p=p, n=self.samples, random_state=self._rng)
                self.scene_patch_generators[s] = sampler()  # Store generator


    def _init_patches(self) -> None:
        """ Initialize the pool of images patches for this dataset. """
        if hasattr(self, 'samples'):
            self.patches = self._sample_patches(self.samples, beta=10.0)


    def _sample_patches(
        self,
        n : int,
        beta : float = 10.0,
    ) -> List[Tuple[str, Tuple[slice, slice]]]:
        """
        Samples n patches with randomized windows.

        Args:
            n (int): Number of samples.
            beta (float): Inverse-temperature for the exponential distribution.

        Returns:
            A list of tuples, each with a scene name and a window (tuple of slices).
        """

        ## Samples scenes.
        if self.only_whole_scenes:
            self._window_size = None
            self.samples = len(self.scenes)
            scenes = self.scenes
        else:
            self._rng.set_state(self._random_state)  # Reset the RNG state
            C = [self.scene_clearances[s] for s in self.scenes]
            idx = rand_index_exponential(C, n=n, beta=beta, replace=True, random_state=self._rng)
            scenes = self.scenes[idx]

        ## Sample windows.
        if self.xarrays:
            self._rng.set_state(self._random_state)  # Reset the RNG state
            ts = {s : self.xarrays[s].shape[2:] for s in scenes}
            if self.sample_by_clearance:
                windows = [self._random_window(self.window_size, scene=s) for s in scenes]
            else:
                windows = [self._random_window(self.window_size, tile_size=ts[s]) for s in scenes]
        else:
            windows = [(slice(None), slice(None)),] * len(scenes)

        return list(zip(scenes, windows))


    def _random_window(
        self,
        window_size : Optional[int],
        tile_size : Optional[Tuple[int, int]] = None,
        scene: Optional[str] = None,
    ) -> Tuple[slice, slice]:
        """
        Random crop window, according to window and tile size.

        Args:
            window_size (int): Size of the square window.
            tile_size ((int, int), optional): Size of the tile to sample from.
            p (np.ndarray, optional): 2d probability array (sums to 1).
                The top-left corner of a patch of size `window_size` is
                sampled with probability `p`. Overrides `tile_size`.

        Returns:
            Window of the form (slice(left, right), slice(top, bottom)).
        """

        ws, ts = window_size, tile_size

        if self.sample_by_clearance and scene is not None and ws:
            generator = self.scene_patch_generators[scene]
            yi, xi = next(generator)
            return (slice(yi, yi + ws), slice(xi, xi + ws))
        elif ts and ws:
            assert ts[0] >= ws, f"Tile is smaller than the window: {ts[0]} < {ws}."
            assert ts[1] >= ws, f"Tile is smaller than the window: {ts[1]} < {ws}."
            if self.split_scene_mode == 'within':
                ## Sample from the train / val / test portion of the scene
                if self.train_val_test == 'train':
                    ylow, yhigh = 0, int(ts[0] * 0.8) - ws
                    xlow, xhigh = 0, ts[1] - ws
                elif self.train_val_test == 'val':
                    ylow, yhigh = int(ts[0] * 0.8), ts[0] - ws
                    xlow, xhigh = 0, int(ts[1] * 0.5) - ws
                elif self.train_val_test == 'test':
                    ylow, yhigh = int(ts[0] * 0.8), ts[0] - ws
                    xlow, xhigh = int(ts[1] * 0.5), ts[1] - ws
                else:
                    ylow, yhigh = 0, ts[0] - ws
                    xlow, xhigh = 0, ts[1] - ws
            elif self.split_scene_mode == 'across':
                ylow, yhigh = 0, ts[0] - ws
                xlow, xhigh = 0, ts[1] - ws
            yi = self._rng.randint(ylow, yhigh)  # Sample from [low, high)
            xi = self._rng.randint(xlow, xhigh)
            return (slice(yi, yi + ws), slice(xi, xi + ws))
        else:
            return (slice(None), slice(None))


    def __len__(self) -> int:
        """ Dataset length: number of patches sampled from all scenes. """
        return len(self.patches)


    def __getitem__(
        self,
        idx : Tuple[Union[int, str], Optional[slice], Optional[slice]],
    ) -> Item:
        """
        Default getter, that uses an index on `self.patches`.

        Args:
            idx: a numerical or str index, with optional slicing. See Usage below.

        Returns:
            The item dictionary.

        Usage:
            self[0]  # Gets the data from scene and window in self.patches[0]
            self[0, :100, 200:300]  # Overrides the window in self.patches[0]
            self['scene-1']  # Gets all the data from of scene-1
            self['scene-1', :100, 200:300]  # Same but with a 100x100 window
        """

        window = None
        if isinstance(idx, tuple):  # Given window overrides patch window
            idx, window = idx[0], (idx[1], idx[2])
        if isinstance(idx, int):
            scene, window_ = self.patches[idx]
        elif isinstance(idx, str):
            scene, window_ = idx, (slice(None), slice(None))

        window = window or window_
        labels = []
        if self.labels:
            images, clouds, labels, dates = self._getscene(scene, window)
        else:
            images, clouds, dates = self._getscene(scene, window)

        clear_masks = self.clearance_masks(clouds)
        clearances = torch.from_numpy(np.array(clear_masks).mean(axis=(1, 2, 3)))

        item = Item({
            'images': images,  # (T, C, H, W)
            'dates': dates,
            'clouds': clouds,  # (T, C, H, W)
            'clearances': clearances,
            'scene': scene,
            'revisits': int(images.shape[0]),  # Useful for masking
        })

        if self.labels:
            item.update({'labels':labels})

        return item


    def _getscene(
        self,
        scene : str,
        window : Optional[Tuple[slice, slice]] = None,
    ) -> Tuple[Array, Array, np.ndarray]:
        """
        Getter that uses a string index on `self.scenes`, with an optional patch window.

        Args:
            idx (str): String index.
            window ((slice, slice), optional): slicing. See Usage below.

        Usage:
            self._getscene('scene-1')  # Gets all the data from scene-1
            self._getscene('scene-1', (slice(0, 30), slice(0, 30)))  # Same but with a 30x30 window
        """

        if self.sat is None:
            raise NotImplementedError('Getter for this class is not implemented.')

        window = window or (slice(None), slice(None))
        images = self.xarrays[scene][:, :, window[0], window[1]]
        clouds = self.xarrays_cloud[scene].expand_dims(dim='band', axis=1)[:, :, window[0], window[1]]

        if self.labels:
            labels = self.xarrays_labels[scene].expand_dims(dim='band', axis=1)[:, :, window[0], window[1]]
            labels = torch.from_numpy(labels.values.copy())  # Copy from a read-only array

        ## Datetimes
        dates = images['time'].to_index().date

        ## Transform images.
        if self.transform:
            images = self.transform(images)
            if isinstance(images, torch.Tensor):
                clouds = torch.from_numpy(clouds.values.copy())  # Copy from a read-only array

        if self.labels:
            return (images, clouds, labels, dates)
        else:
            return (images, clouds, dates)


    def __str__(self) -> str:
        if self.date_range is not None:
            date_str = f"start {self.date_range.min().date()}, end {self.date_range.max().date()}"
        else:
            date_str = "start na, end na"
        return (
            f"split {self.train_val_test}\n"
            f"sat {self.sat}\n"
            f"{date_str}\n"
            f"scenes {len(self.scenes)}, samples {len(self)}\n"
            f"labels {self.labels}\n"
            f"window_size {self.window_size}\n"
            f"transform {self.transform}\n"
            f"random_seed {self.random_seed}\n"
            f"data_dir {self.data_dir}\n"
            f"band means {[round(float(m), 2) for m in self.mean]}\n"
            f"band stds {[round(float(m), 2) for m in self.std]}"
        )



class WotusDataset(SatelliteDataset):
    """
    WOTUS dataset.
    """

    mean = {'planet': np.array([434, 674, 717, 2216]),
            'sentinel': np.array([2486, 2308, 2191, 2197, 2435, 3061, 3325, 3264, 3493, 1812,  389, 2224, 1588])}
    std = {'planet': np.array([187, 237, 325, 563]),
           'sentinel': np.array([ 998, 1052, 1037, 1152, 1148, 1143, 1176, 1175, 1194,  856,  349, 898, 761])}
    MNT = MNT_WOTUS
    index = wotus_index
    default_bands = {'planet':list(range(4)), 'sentinel':list(range(13))}

    def __init__(self, *args, **kwargs) -> None:
        df = __class__.index(trigger_download_if_not_exists=kwargs.get('download', False), data_dir=__class__.MNT)
        sat = kwargs['sat']
        if sat not in ['planet', 'sentinel']:
            raise ValueError(f"Expected sat value: 'planet'|'sentinel'. Got {sat}.")
        kwargs.update({'data_dir':__class__.MNT, 'download':None, 'train':None, 'df':df,
                       'bands':kwargs.get('bands', __class__.default_bands[sat]), 'sat':kwargs.get('sat', sat)})
        self.mean = __class__.mean[sat][kwargs['bands']]
        self.std = __class__.std[sat][kwargs['bands']]
        super().__init__(*args, **kwargs)


    def clearance_masks(self, cloud_masks : xr.DataArray, **kwargs : Any) -> xr.DataArray:
        if self.sat == 'sentinel':
            ## TODO: for some reason, S2 cloud masks can be uint8, like thos of SN7.
            if np.array(cloud_masks).max() > 2:  # cloud_masks.dtype.type is np.uint8:
                return ((cloud_masks != 9) & (cloud_masks > 1))
            else:
                return (cloud_masks >= 1) & (cloud_masks <= 1.5)  # 0: invalid, (>=1 & <=1.5): clear (>1.5): cloud
        elif self.sat == 'planet':
            return (cloud_masks != 1)

    def cloud_masks(self, images : List[xr.DataArray], **kwargs : Any) -> List[xr.DataArray]:
        if self.sat == 'sentinel':
            return [x[-1] for x in images]
        elif self.sat == 'planet':
            zeros = xr.zeros_like(images[0][-1])
            return [zeros for _ in range(len(images))]



class SN7Dataset(SatelliteDataset):
    """
    SpaceNet7 dataset.
    """

    mean = {'planet': np.array([120, 105, 77]) / 255,
            'sentinel': np.array([1972, 1989, 2137, 2245, 2514, 2825, 2962, 3048, 3047, 3740, 2422, 2017])}
    std = {'planet': np.array([40, 32, 29]) / 255,
           'sentinel': np.array([471, 517, 522, 576, 565, 590, 617, 674, 640, 766, 596, 561])}
    MNT = MNT_SPACENET7
    split_json = TRAIN_VAL_TEST_SPLIT_SN7  # Path of the JSON file with train/val/test split data.
    index = spacenet7_index
    default_bands = {'planet':list(range(3)), 'sentinel':list(range(12))}

    def __init__(self, *args, **kwargs) -> None:
        df = __class__.index(trigger_download_if_not_exists=kwargs.get('download', False),
                             data_dir=__class__.MNT, split_json=__class__.split_json, random_split=False)
        sat = kwargs['sat']
        if sat not in ['planet', 'sentinel']:
            raise ValueError(f"Expected sat value: 'planet'|'sentinel'. Got {sat}.")
        kwargs.update({'data_dir':__class__.MNT, 'download':None, 'train':None, 'df':df,
                       'bands':kwargs.get('bands', __class__.default_bands[sat]), 'sat':kwargs.get('sat', sat)})
        self.mean = __class__.mean[sat][kwargs['bands']]
        self.std = __class__.std[sat][kwargs['bands']]
        super().__init__(*args, **kwargs)


    def clearance_masks(self, cloud_masks : xr.DataArray, **kwargs : Any) -> xr.DataArray:
        if self.sat == 'sentinel':
            return ((cloud_masks != 9) & (cloud_masks > 1))
        elif self.sat == 'planet':
            return (cloud_masks != 1)

    def cloud_masks(self, scene : Optional[str], images : Optional[List[xr.DataArray]]) -> List[xr.DataArray]:
        if self.sat == 'sentinel':
            return [x[-1] for x in images]
        elif self.sat == 'planet':
            paths_cloud = self.df.loc[(self.sat, scene)]['cloud_mask_path']
            return [xr.open_rasterio(p)[0] for p in paths_cloud]



class ConcatSatelliteDataset(Dataset):
    """
    Dataset of two SpaceNet7 datasets with synced getters.
    """

    def __init__(
        self,
        datasets : Tuple[Dataset, Dataset],
        window_size_planet : Optional[int] = None,
        window_size_s2 : Optional[int] = None,
        transform : Optional[Callable] = None,
        force_size : Optional[int] = None,
        random_seed : int = DEFAULT_SEED,
    ) -> None:
        """
        Args:
            datasets (tuple of Dataset): PlanetScope and Sentinel datasets.
            window_size_planet (int, optional): Size of the Planet cropping window.
            window_size_s2 (int, optional): Size of the Sentinel cropping window.
                At most one of `window_size_planet` and `window_size_s2` can be defined.
            transform (callable, optional): Transform to be applied on an item.
            force_size (int, optional): If `window_size_s2` is used, this forces the size
                of the Planet windows. If `window_size_planet` is used, this forces the size
                of the Sentinel windows.
            random_seed (int, default={datasets.DEFAULT_SEED}): For the reproducible sampling from datasets.
        """

        if window_size_planet and window_size_s2:
            raise(ValueError("At most one of `window_size_planet` and `window_size_s2` can be defined."))
        self.window_size_planet = window_size_planet
        self.window_size_s2 = window_size_s2
        self.transform = transform

        P, S = self.dataset_planet, self.dataset_sentinel = datasets
        ## Dataset transforms to be overriden by `self.transform`.
        if self.transform:
            P.transform, S.transform = None, None
        self.random_seed = random_seed

        # Reset the RNG states.
        P.random_seed, S.random_seed = self.random_seed, self.random_seed

        ## Common scenes.
        common = set(P.scenes) & set(S.scenes)
        ## Init scenes to the new overlap, and reset patches.
        ix_common = S.df.index.get_level_values('scene').isin(common)
        S.df = S.df.loc[ix_common]
        S._init_scenes()
        S._init_xarrays()
        S._init_clearances()
        S._init_scene_patch_generators()
        S._init_patches()  # Patch windows will be None

        ## Sample patches.
        S._init_scene_patch_generators()
        P._init_scene_patch_generators()
        P.patches = S.patches.copy()

        if not (self.window_size_s2 or self.window_size_planet):
            win_p = (slice(None), slice(None))
            win_s = (slice(None), slice(None))

        for i, (scene, _) in enumerate(S.patches):
            ts_p = P.xarrays[scene].shape[2:]  # Tile sizes
            ts_s = S.xarrays[scene].shape[2:]
            if window_size_s2:
                gsd_ratio = tuple(np.array(ts_p) / np.array(ts_s))
                win_s = S._random_window(window_size_s2, tile_size=ts_s)
                win_p = scale_window(win_s, gsd_ratio, force_size=force_size)
            elif window_size_planet:
                gsd_ratio = tuple(np.array(ts_s) / np.array(ts_p))
                win_p = P._random_window(window_size_planet, tile_size=ts_p)
                win_s = scale_window(win_p, gsd_ratio, force_size=force_size)

            P.patches[i] = (scene, win_p)
            S.patches[i] = (scene, win_s)


    def __len__(self) -> int:
        return len(self.dataset_sentinel)


    def __getitem__(self, idx : Union[int, str]) -> Dict[str, Item]:
        """ Get paired items. """

        P, S = self.dataset_planet, self.dataset_sentinel
        if isinstance(idx, str):
            scene = idx
            win_p = win_s = (slice(None), slice(None))
        elif isinstance(idx, int):
            (scene, win_p), (_, win_s) = P.patches[idx], S.patches[idx]

        item = {
            'highres': P[scene, win_p[0], win_p[1]],
            'lowres': S[scene, win_s[0], win_s[1]],
        }

        return self.transform(item) if self.transform else item
