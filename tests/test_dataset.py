import numpy as np
import pytest
import torch
from torchvision import transforms


from src.datasets import *
from src.datasources import *
from src.datautils import *
from src.datamodules import *


date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")

dataset_planet = SN7Dataset(sat='planet',
                            date_range=date_range,
                            samples=100,
                            window_size=100,
                            transform=Compose4d([ToTensor2()]),
                            random_seed=42, debug=True,
                            sample_by_clearance=True)

dataset_planet_whole_scenes = SN7Dataset(sat='planet',
                                         date_range=date_range,
                                         only_whole_scenes=True,
                                         transform=Compose4d([ToTensor2()]),
                                         random_seed=42, debug=True)

dataset_sentinel = SN7Dataset(sat='sentinel',
                              date_range=date_range,
                              samples=100,
                              window_size=100,
                              transform=Compose4d([ToTensor2()]),
                              random_seed=42, debug=True)

dataset_concat = ConcatSatelliteDataset((dataset_planet, dataset_sentinel),
                                        window_size_planet=100,
                                        random_seed=42)


def test_Item_repr():
    print(dataset_planet[0])
    print(dataset_sentinel[0])


def test_SatelliteDataset_NotImplemented():
    with pytest.raises(NotImplementedError):
        super(SN7Dataset, dataset_planet).cloud_masks()

    with pytest.raises(NotImplementedError):
        super(SN7Dataset, dataset_planet).clearance_masks()


def test_SatelliteDataset():
    n = 100
    assert len(dataset_planet.patches) == n
    assert len(dataset_planet.patches) == len(dataset_planet)


def test_SatelliteDataset_no_scenes_warning():
    with pytest.warns(UserWarning):
        ## Invalid year!
        SN7Dataset(sat='planet', date_range=pd.date_range(start=f"2021-01-01", end=f"2021-01-31"))


def test_SatelliteDataset_split_scene_mode():
    with pytest.raises(ValueError):
        SN7Dataset(sat='planet', date_range=date_range, split_scene_mode='bad value')

    ## Within
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='train', split_scene_mode='within', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='val', split_scene_mode='within', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='test', split_scene_mode='within', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test=None, split_scene_mode='within', debug=True)

    ## Across
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='train', split_scene_mode='across', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='val', split_scene_mode='across', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test='test', split_scene_mode='across', debug=True)
    SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=100,
               train_val_test=None, split_scene_mode='across', debug=True)


def test_SatelliteDataset_init_scenes():
    dataset_planet.train_val_test = 'train'
    dataset_planet._init_scenes()

    with pytest.raises(ValueError):
        dataset_planet.train_val_test = 'bad value'
        dataset_planet._init_scenes()
    dataset_planet.train_val_test = None

    with pytest.raises(TypeError):  # date_range is filtered in the init() of the SpaceNet7Dataset class
        dataset_planet.date_range = 'bad value'
        dataset_planet._init_scenes()
    dataset_planet.date_range = date_range


def test_SatelliteDataset_init_xarrays():
    dataset_planet.date_range = None
    dataset_planet._init_xarrays()
    dataset_planet.date_range = date_range


def test_SatelliteDataset_random_seed():
    dataset_planet.random_seed = 42
    seed = dataset_planet.random_seed
    assert seed == 42
    assert dataset_planet._rng is not None
    assert dataset_planet._random_state is not None


def test_SatelliteDataset_window_size():
    dataset_planet.window_size = 100  # Test setter
    with pytest.raises(TypeError):
        dataset_planet.window_size = 'bad value'


def test_SatelliteDataset_only_whole_scenes():
    assert len(dataset_planet_whole_scenes) == len(dataset_planet_whole_scenes.scenes)
    assert dataset_planet_whole_scenes.window_size is None


def test_SatelliteDataset_clearance_probability():
    clearance_masks = np.zeros((5, 5), dtype=bool)
    p = dataset_planet._clearance_probability(clearance_masks)
    assert p.sum() == 1


def test_SatelliteDataset_random_window():
    dataset_planet._random_window(window_size=100, tile_size=(500, 500))
    scene, window = dataset_planet.patches[0]
    assert scene  # scene is not '' or None
    assert window[0].start  # window slice is not None
    assert dataset_planet._random_window(None, None) == (slice(None), slice(None))


def test_SatelliteDataset_unique_patches():
    patches = [(p[0], str(p[1])) for p in dataset_sentinel.patches]
    unique_patches = set([(p[0], str(p[1])) for p in dataset_sentinel.patches])
    assert len(patches) - len(unique_patches) <= 1


def test_SatelliteDataset_getter_str_input():
    scene = dataset_planet.scenes[0]  # 'L15-0331E-1257N_1327_3160_13'
    dataset_planet[scene]
    assert len(dataset_planet._getscene(scene)[0]) == len(dataset_planet[scene]['images'])

    images = dataset_planet[scene, :100, :100]['images']
    assert images.shape[2:] == (100, 100)

    dataset_planet.sat = None
    with pytest.raises(NotImplementedError):
        dataset_planet[scene]
    dataset_planet.sat = 'planet'


def test_SatelliteDataset_getter_int():
    idx = 0
    assert dataset_planet[idx]['images'].ndim == 4
    images = dataset_planet[idx, :50, :30]['images']

    assert images.shape[2:] == (50, 30)

    ## Tests transforms that result in numpy arrays.
    dataset_planet.transform = transforms.Compose([])
    assert dataset_planet[idx]['images'].shape == (1, 3, 100, 100)


def test_SatelliteDataset_cloud_dtype():
    idx = 0
    dataset_planet.transform = Compose4d([ToTensor2()])
    clouds = dataset_planet[idx]['clouds']
    assert clouds.dtype is torch.uint8


def test_SatelliteDataset_labels():
    dataset_planet.labels = False
    dataset_planet._init_xarrays()
    assert 'labels' not in dataset_planet[0]

    dataset_planet.labels = True
    dataset_planet._init_xarrays()
    item = dataset_planet[0]
    assert 'labels' in item
    assert item['labels'].shape == (1, 1, 100, 100)


def test_SatelliteDataset_repr():
    dataset_planet.date_range = None
    print(dataset_planet)
    dataset_planet.date_range = date_range
    print(dataset_planet)


def test_SN7Dataset():
    date_range = pd.date_range(start=f"2019-12-30", end=f"2020-01-31")
    dataset = SN7Dataset(sat='planet', date_range=date_range, samples=100, window_size=50,
                           transform=Compose4d([ToTensor2()]), debug=True)
    assert len(dataset.scenes) > 0
    assert len(dataset.patches) > 0

    dataset = SN7Dataset(sat='sentinel', date_range=date_range, samples=100, window_size=50,
                           transform=Compose4d([ToTensor2()]), debug=True)
    assert len(dataset.scenes) > 0
    assert len(dataset.patches) > 0

    with pytest.raises(ValueError):
        dataset = SN7Dataset(sat='bad value')


def test_WotusDataset():
    date_range = pd.date_range(start=f"2018-08-30", end=f"2019-10-31")
    dataset = WotusDataset(sat='planet', date_range=date_range, samples=100, window_size=100,
                           transform=Compose4d([ToTensor2()]), debug=True)
    assert len(dataset.scenes) > 0
    assert len(dataset.patches) > 0

    dataset = WotusDataset(sat='sentinel', date_range=date_range, samples=100, window_size=100,
                           transform=Compose4d([ToTensor2()]), debug=True)
    assert len(dataset.scenes) > 0
    assert len(dataset.patches) > 0

    with pytest.raises(ValueError):
        dataset = WotusDataset(sat='bad value')


def test_WotusDataset_clearance_masks():
    date_range = pd.date_range(start=f"2018-08-30", end=f"2019-10-31")
    dataset = WotusDataset(sat='sentinel', date_range=date_range, samples=100, window_size=100,
                           transform=Compose4d([ToTensor2()]), debug=True)
    assert dataset.clearance_masks(dataset[0]['clouds']).shape == dataset[0]['clouds'].shape

    assert dataset.clearance_masks(10 * np.random.rand(5, 5)).shape == (5, 5)


def test_SpaceNet7ConcatDataset_init():
    with pytest.raises(ValueError):
        ConcatSatelliteDataset((dataset_planet, dataset_sentinel), window_size_planet=134, window_size_s2=64)


def test_SpaceNet7ConcatDataset_getter_int():
    images = dataset_concat[0]['highres']['images']
    assert images.shape[2:] == (100, 100)


def test_SpaceNet7ConcatDataset_getter_str():
    scene = dataset_concat.dataset_sentinel.scenes[0]
    images = dataset_concat[scene]['highres']['images']


def test_SpaceNet7ConcatDataset_len():
    assert len(dataset_concat) == 100


def test_SpaceNet7ConcatDataset_window_size():
    dataset = ConcatSatelliteDataset((dataset_planet, dataset_sentinel))#, window_size_planet=None, window_size_s2=None)
    assert dataset.dataset_planet.patches[0][1] == (slice(None), slice(None))
    
    dataset = ConcatSatelliteDataset((dataset_planet, dataset_sentinel), window_size_planet=100)#, window_size_s2=None)
    assert dataset[0]['highres']['images'].shape[-2:] == (100, 100)

    dataset = ConcatSatelliteDataset((dataset_planet, dataset_sentinel), window_size_s2=50)
    assert dataset[0]['lowres']['images'].shape[-2:] == (50, 50)


def test_SpaceNet7ConcatDataset_transform():
    ConcatSatelliteDataset((dataset_planet, dataset_sentinel), window_size_planet=100, transform=lambda item: item)
