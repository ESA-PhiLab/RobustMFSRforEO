from typing import List, Tuple, Dict, Union, Optional, Callable, Iterable

import cv2
import numpy as np
from numpy.random import RandomState
import torch
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose, ToTensor
import xarray as xr
import math

Array = Union[Tensor, np.ndarray, xr.DataArray]


class Sampler2d:
    """
    Generator for a 2D probability sampler.
    """
    def __init__(
        self,
        p : np.ndarray,
        n : int,
        random_state : Optional[RandomState] = None,
    ) -> None:
        """
        Random sampler from a 2D probability array.

        Args:
            p (np.ndarray): 2D probability array.
            n (int) : number of samples.
            random_state (RandomState, optional): initial RNG state.
        """

        if p.sum() == 0:
            p[:] = 1 / p.size
        else:
            p = p / p.sum()

        self.n = n
        yx = np.array(np.meshgrid(*(range(s) for s in p.shape))).reshape(2, -1)
        rng = random_state or RandomState(seed=1337)
        idx = rng.choice(range(p.size), size=n, p=p.flatten(order='F'))
        self.yi, self.xi = yx[:, idx]
        self.yi, self.xi = list(self.yi), list(self.xi)

    def __call__(self):
        for i in range(self.n):
            yield self.yi[i], self.xi[i]


def dilate(x : np.ndarray, iterations : int) -> np.ndarray:
    """ Dilate binary mask. """
    shape, type_ = x.shape, x.dtype
    H, W = shape[-2:]
    K = np.ones((3, 3), dtype=np.uint8)  # Kernel
    x = x.astype(np.uint8).reshape(-1, H, W).transpose(1, 2, 0)
    x = cv2.dilate(x, K, iterations=iterations).reshape(H, W, -1)
    return x.transpose(2, 0, 1).reshape(*shape).astype(type_)


def scale_window(
    window : Union[Tuple[slice, slice], Tuple[int, int]],
    s : Tuple[float, float],
    force_size : Optional[int] = None,
) -> Tuple[slice, slice]:
    """
    Rescale window by a factor of `s`.
    """

    if not (isinstance(s, tuple) and len(s) == 2):
        raise TypeError(f'Expected types for scaling factor: float, or (float, float). Got {s}.')

    if isinstance(window, tuple) and len(window) == 2:
        x_slice, y_slice = window
        
        if isinstance(x_slice, slice) and isinstance(y_slice, slice):
            l, r, t, b = x_slice.start, x_slice.stop, y_slice.start, y_slice.stop
            l2, r2 = [int(s[0] * x) if x is not None else None for x in (l, r)]
            t2, b2 = [int(s[1] * y) if y is not None else None for y in (t, b)]
            if force_size is not None:
                r2 = l2 + force_size if l2 else force_size
                b2 = t2 + force_size if t2 else force_size
            return (slice(l2, r2), slice(t2, b2))

        elif isinstance(x_slice, int) and isinstance(y_slice, int):
            return tuple([int(s[i] * w) if w else None for i, w in enumerate(window)])

    raise TypeError(f'window expected type: (slice, slice) or (int, int). Got {window}.')


def train_val_test_partition(
    x : list,
    split : Tuple[float, float, float],
    seed : int = 1337,
) -> dict:
    """
    Partition `x` into training, validation, test sets.

    Args:
        x (list): a list of unique elements.
        split ((float, float, float): tuple of splitting proportions.
            Must sum to one.
        seed (int, optional): random seed.

    Returns:
        A dictionary that maps a each unique element in x to 'train',
            'val' or 'test'.
    """

    if len(set(x)) != len(x): raise ValueError('x must contain unique values.')
    if sum(split) != 1: raise ValueError('split values must sum to one.')
    lengths = [round(len(x) * s) for s in split]
    lengths[0] -= sum(lengths) - len(x)
    parts = random_split(x, lengths=lengths, generator=torch.Generator().manual_seed(42))
    return {elem : class_
            for part, class_ in zip(parts, ('train', 'val', 'test'))
            for elem in part}


def rand_index_exponential(
    x : Union[np.ndarray, List[float]],
    n : Optional[int] = None,
    beta : float = 50.0,
    replace : bool = True,
    random_state : Optional[RandomState] = None,
) -> np.ndarray:
    """
    Samples `n` indices with probability exponentially proportional to `x`.

    Args:
        x (numpy.ndarray): Exponential proportionality values.
        n (int, optional): Number of samples.
        beta (float, optional): Inverse-temperature parameter of the exponential
            distribution.
            beta = 1e-3 practically amounts to uniform sampling.
            beta = 1e6 practically amounts to the argmax.
        random_state (RandomState, optional): NumPy random number generator.

    Returns:
        (numpy.ndarray): Sampled indices.
    """

    if len(x) == 0: return []
    x = np.array(x)
    rng = random_state or RandomState(seed=1337)
    e = np.exp(beta * x / (x.max() + 1e-5))
    p = e / e.sum()
    idx = range(len(p))

    return rng.choice(idx, size=n, p=p, replace=replace)



class Clamp:
    """
    Transform for torch.clamp.
    """
    def __init__(self, min : float, max : float) -> None:
        self.min, self.max = min, max
    def __call__(self, x : Tensor) -> Tensor:
        return torch.clamp(x, min=self.min, max=self.max)



class ToTensor2(ToTensor):
    """
    ToTensor wrapper that expects (C, H, W) and outputs (C, H, W).
    """
    def __init__(self, normalize : bool = True) -> None:
        self.normalize = normalize
    def __call__(self, x : Union[np.ndarray, xr.DataArray]) -> Tensor:
        if isinstance(x, xr.DataArray):
            x = x.values.copy()
        ## ToTensor expect PIL-like dimension order (H x W x C).
        if not self.normalize:
            x = x.astype(np.int32)
        return super().__call__(x.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)



class Compose4d(Compose):
    """
    Compose :class:`~torchvision.transforms` for a batch of images.
    """
    def __call__(self, x : Array) -> Array:
        assert x[0].ndim == 3
        return self._transform4d(super().__call__, x)

    def _transform4d(self, t : Callable, x : List[Array]) -> Array:
        """ List of 3D -->  4D """
        x = [t(img) for img in x]
        dtype, cls = x[0].dtype, type(x[0])
        ## Stack method is type-dependent.
        if cls is np.ndarray: return np.array(x, dtype=dtype)
        elif cls is Tensor: return torch.stack(x).to(dtype)
        elif cls is xr.DataArray: return xr.concat(x, dim='t').astype(dtype)
        else: raise TypeError(f"Unsupported type {cls}.")



class PadCollate:
    """
    A variant of collate_fn that pads according to the longest sequence in a batch of sequences.

    Based on
    - https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
    - https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    """

    def __init__(self, dim : Union[int, Tuple[int, ...]] = 0, powerOf2 : bool = False):
        """
        Args:
            dim (int): the dimension to be padded (dimension of time in sequences).
            powerOf2 (bool): pad to the least upper bound to the len of the longest sequence that is a power of 2.
        """
        self.dim = [dim] if isinstance(dim, (int, np.int64, np.int32)) else list(dim)
        self.powerOf2 = powerOf2

    def _pad_tensor(
        self,
        vec : Tensor,
        pad : Union[int, List[int]],
        dim : Union[int, List[int]],
    ) -> Tensor:
        """
        Args:
            vec (Tensor): tensor to pad.
            pad (int, list(int)): the size to pad to. Must be the same size as `dim`.
            dim (int, list(int)): dimensions to pad. Must be the same size as `pad`.

        Returns:
            A new tensor padded to `pad` in dimension `dim`.
        """
        dim = [dim] if isinstance(dim, (int, np.int64, np.int32)) else list(dim)
        pad = [pad] if isinstance(pad, (int, np.int64, np.int32)) else list(pad)
        assert len(dim) == len(pad)
        for d, p in zip(dim, pad):
            pad_size = list(vec.shape)
            pad_size[d] = p - vec.shape[d]
            vec = torch.cat([vec, torch.zeros(*pad_size)], dim=d)
        return vec


    def pad_collate(
        self,
        batch : List[Tensor],
        dim : Optional[Union[int, List[int]]] = None,
    ) -> Tensor:
        """
        Args:
            batch (list): list of tensors.
            dim (int, list(int)): dimensions to pad. Overrides `self.dim`.

        Returns:
            A 5D tensor (B, T, C, H, W), with padded 4D tensors from `batch`.
        """
        dim = self.dim if dim is None else dim
        dim = [dim] if isinstance(dim, (int, np.int64, np.int32)) else list(dim)
        ## Find longest sequence in either dimension.
        max_len =  np.array([x.shape for x in batch], dtype=int).max(axis=0)
        ## Get max len power of 2 for dim 0
        if self.powerOf2:
            max_len[0] = calculateNextPowerOf2(max_len[0])
        ## Pad according to max_len.
        batch = [self._pad_tensor(x, pad=max_len[dim], dim=dim) for x in batch]
        ## Stack all.
        return torch.stack(batch, dim=0)


    def __call__(self, batch : List[Dict]) -> Dict:
        images = self.pad_collate([x['images'] for x in batch], dim=(0, 2, 3))
        num_revisits = [int(x['images'].shape[0]) for x in batch]
        max_revisits = images.shape[1]
        revisits_indicator = np.zeros((len(num_revisits),max_revisits))
        for i, r in enumerate(num_revisits):
            revisits_indicator[i, :r] = 1
        item = {
            ## Pad images and clouds on time, y, x dimension.
            'images': images,
            'clouds': self.pad_collate([x['clouds'] for x in batch], dim=(0, 2, 3)),
            'clearances': self.pad_collate([x['clearances'] for x in batch], dim=0),
            'scene': [x['scene'] for x in batch],
            'revisits_indicator': torch.from_numpy(revisits_indicator.copy()),  # Useful for masking
        }

        if 'labels' in batch[0]:
            labels = self.pad_collate([x['labels'] for x in batch], dim=(0, 2, 3))
            item.update({'labels':labels})

        return item



class ConcatPadCollate:
    def __call__(self, batch : List[Dict[str, Dict]]) -> Dict:
        collate_fn = PadCollate()
        elem = batch[0]
        return {key: collate_fn([x[key] for x in batch]) for key in elem}



class ConcatPadPTwoCollate:
    def __call__(self, batch : List[Dict[str, Dict]]) -> Dict:
        collate_fn = PadCollate(powerOf2=True)
        elem = batch[0]
        return {key: collate_fn([x[key] for x in batch]) for key in elem}    


def calculateNextPowerOf2(num : int) -> int:
    """
    Args:
        num: int, number
    Returns:
        next power of 2: int, next power of 2
    """
    nextpowerof2 = math.ceil(math.log(num,2))
    powerof2 = int(math.pow(2, nextpowerof2))
    return powerof2
