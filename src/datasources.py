from collections import OrderedDict
from datetime import datetime
import os
import os.path as osp
import re
import subprocess
from typing import Tuple, Optional

from glob import glob
import pandas as pd

from src.datautils import train_val_test_partition


MNT_SPACENET6 = '/data/spacenet6/train/'
PATH_BUCKET_SPACENET6 = "gs://fdl_srhallucinate/spacenet6/train/"

MNT_SPACENET7 = '/data/spacenet/train/'
PATH_BUCKET_SPACENET7 = "gs://fdl_srhallucinate/spacenet/train/"

MNT_WOTUS = '/data/wotus/train'
PATH_BUCKET_WOTUS = "gs://fdl_srhallucinate/wotus/train/"

TRAIN_VAL_TEST_SPLIT_SN7 = 'src/scene_split2.json'


## Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/composites/
S2_BANDS = OrderedDict({
    'B01':'aerosol',
    'B02':'blue',
    'B03':'green',
    'B04':'red',
    'B05':'red_edge_1',
    'B06':'red_edge_2',
    'B07':'red_edge_3',
    'B08':'nir_1',
    'B8A':'nir_2',
    'B09':'water_vapour',
    ## 'B10' ('cirrus' detector) is missing
    'B11':'swir_1',
    'B12':'swir_2',
})
S2_BANDS12 = S2_BANDS
S2_BANDS13 = S2_BANDS.copy()
S2_BANDS13['B10'] = 'cirrus'
S2_BANDS.update({S2_BANDS12[v]:i
                 for i, v in enumerate(['B01', 'B02', 'B03', 'B04',
                                        'B05', 'B06', 'B07', 'B08',
                                        'B8A', 'B09', 'B11', 'B12'])})
S2_BANDS13.update({S2_BANDS13[v]:i
                   for i, v in enumerate(['B01', 'B02', 'B03', 'B04',
                                          'B05', 'B06', 'B07', 'B08',
                                          'B8A', 'B09', 'B10', 'B11', 'B12'])})
for S2B in [S2_BANDS12, S2_BANDS13]:
    S2B['true_color'] = [S2B[S2B[n]] for n in ("B04", "B03", "B02")]
    S2B['false_color'] = [S2B[S2B[n]] for n in ("B08", "B04", "B03")]
    S2B['swir'] = [S2B[S2B[n]] for n in ("B12", "B08", "B04")]
    S2B['agriculture'] = [S2B[S2B[n]] for n in ("B11", "B08", "B02")]
    S2B['geology'] = [S2B[S2B[n]] for n in ("B12", "B11", "B02")]
    S2B['bathimetric'] = [S2B[S2B[n]] for n in ("B04", "B03", "B01")]

PS_BANDS = OrderedDict({
    'B01':'blue', 'B02':'green', 'B03':'red', 'B04':'nir',
    'blue':0, 'green':1, 'red':2, 'nir':3,
})
PS_BANDS['true_color'] = [PS_BANDS[n] for n in ('red', 'green', 'blue')]

## TODO
## - [ ] missing test for download_folder


def glob_gsutil(globloc, isdir=False):
    """" Globbing in the mounted location hangs """

    if isdir:
        commands = ["gsutil", "ls", "-d", globloc]
    else:
        commands = ["gsutil", "ls", globloc]

    glob_result = subprocess.check_output(commands)
    return [g for g in glob_result.decode("UTF-8").split('\n') if g != ""]


def download_folder(folder, scenes_path_bucket=None, path_bucket=None, path_local=None):
    """ Download folder from  bucket to local SSD/HD """
    if scenes_path_bucket is None:
        scenes_path_bucket = glob_gsutil(path_bucket)

    print(f"Downloading folder {folder}")
    for scene_bucket in scenes_path_bucket:
        scene_id = os.path.basename(scene_bucket[:-1])  # remove last /
        folder_bucket_or = os.path.join(scene_bucket, folder)
        folder_ssd_dst = os.path.join(path_local, scene_id+"/")
        os.makedirs(folder_ssd_dst, exist_ok=True)
        ret = subprocess.run(["gsutil", "-m", "cp", "-r", folder_bucket_or, folder_ssd_dst])
        if not hasattr(ret, "returncode") or (ret.returncode != 0):
            print(f"{scene_id} missing skip")

    return scenes_path_bucket


def download_folders(folders, force_download, trigger_download_if_not_exists, path_bucket,
                     path_local):
    """
    Loop a list of folders triggering download to local SSD paths.
    if force download it will force the download of the files from bucket otherwise it will download if empty
    """
    scenes_path_bucket = None
    for folder in folders:  # Labels
        if folder is None:
            continue
        glob_str = osp.join(path_local, f'*/{folder}/*')
        paths = sorted(glob(glob_str))
        if force_download or ((len(paths) == 0) and trigger_download_if_not_exists):
            scenes_path_bucket = download_folder(folder, scenes_path_bucket, path_bucket=path_bucket,
                                                 path_local=path_local)

            paths = sorted(glob(glob_str))
        assert len(paths) > 0, f"No data found in folder {folder} {glob_str}"


# Train test split for Wotus made manually to avoid spatial overlap and to have one test image from each AoI
# https://frontierdevelopmentlab.gitlab.io/fdl-us-2020-droughts/xstream/folium_locs_wotus.html
TRAIN_VAL_TEST_SPLIT_WOTUS = {
    "ID_0": "test",
    "ID_1": "train",
    "ID_2": "train",
    "SD_0": "train",
    "SD_1": "train",
    "SD_2": "test",
    "SD_3": "train",
    "SD_Focus": "val",
    "UCEast_0": "val",
    "UCWest_1": "train",
    "UCWest_Focus": "test"
}

def wotus_index(data_dir : str = MNT_WOTUS, folder_s2: str="S2", folder_planet: str ="images",
                trigger_download_if_not_exists: bool=False, force_download: bool=False) -> pd.DataFrame:
    """
     Args:
        data_dir: Data to look up for the scenes
        folder_planet: Name of folder with planet tiffs {MNT_WOTUS}/{scene_id}/{folder_planet}
        folder_s2: Name of folder with Sentinel-2 tiffs {MNT_WOTUS}/{scene_id}/{folder_s2}
        trigger_download_if_not_exists: trigger download from bucket if folder is empty
        force_download:  for re-download from the bucket

     Returns:
        Dataframe with the index of the files. S2 and Planet images are checked that exists
    """
    # Downloads data from bucket and check that all imagery exists
    download_folders([folder_s2, folder_planet],
                     force_download=force_download, trigger_download_if_not_exists=trigger_download_if_not_exists,
                     path_local=data_dir, path_bucket=PATH_BUCKET_WOTUS)

    # These exists because we checked before in the assert
    paths_sentinel = sorted(glob(osp.join(MNT_WOTUS, f'*/{folder_s2}/*')))
    paths_planet = sorted(glob(osp.join(MNT_WOTUS, f'*/{folder_planet}/*')))

    df = pd.concat([pd.DataFrame(dict(path=paths_sentinel, sat='sentinel')),
                    pd.DataFrame(dict(path=paths_planet, sat='planet'))], ignore_index=True)

    ## Scene names, filenames from paths.
    df['scene'] = df.path.map(lambda x: x.split(osp.sep)[4])
    df['basename'] = df['path'].map(lambda x: osp.basename(x))
    df['datetime'] = df['basename'].map(lambda x: datetime.strptime(os.path.splitext(x)[0], "%Y-%m-%d"))
    # Do I need to add year month day?

    df.set_index(['sat', 'scene', 'datetime'], inplace=True)
    df.sort_index(inplace=True)

    df['split'] = df.index.get_level_values('scene').map(TRAIN_VAL_TEST_SPLIT_WOTUS)
    
    return df


def spacenet7_index(
    data_dir : str = MNT_SPACENET7,
    folder_planet : str = "images",
    folder_s2 : str = "S2L2A",
    folder_cloud_mask_planet : str = "UDM_masks",
    folder_cloud_mask_sentinel : Optional[str] = None,
    folder_label_mask_planet : str = "masks",
    trigger_download_if_not_exists : bool = False,
    force_download : bool = False,
    random_split : bool = False,
    train_val_test_split : Tuple[float, float, float] = (.8, .1, .1),
    split_json : Optional[str] = TRAIN_VAL_TEST_SPLIT_SN7,
    random_seed : int = 1337,
) -> pd.DataFrame:
    """
    Creates a dataframe with the file index for the SpaceNet7 problem. It glob files in MNT_SPACENET7 local folder

    Args:
        data_dir: Data to look up for the scenes.
        folder_planet: Name of folder with planet tiffs {MNT_SPACENET7}/{scene_id}/{folder_planet}
        folder_s2: Name of folder with Sentinel-2 tiffs {MNT_SPACENET7}/{scene_id}/{folder_s2}
        folder_cloud_mask_planet: Name of folder with Planet cloud masks {MNT_SPACENET7}/{scene_id}/{folder_cloud_mask_planet}
        folder_cloud_mask_sentinel: Name of folder with Sentinel-2 cloud masks {MNT_SPACENET7}/{scene_id}/{folder_cloud_mask_sentinel}
        folder_label_mask_planet: Name of folder with labels for planet {MNT_SPACENET7}/{scene_id}/{folder_label_mask_planet}
        trigger_download_if_not_exists: trigger download from bucket if folder is empty.
        force_download: for re-download from the bucket.
        random_split: randomly split into train/val/test data.
        train_val_test_split: percentage of train/val/test data.
        split_json: path of the JSON file with train/val/test split data. If provided, train_val_test_split is ignored.
        random_seed: for train/val/test split

    Returns:
        Dataframe with the index of the files. S2 and Planet images are checked that exists

    """

    if random_split:
        assert sum(train_val_test_split) == 1.0
    else:
        assert split_json is not None

    ## Downloads data from bucket and check that all imagery exists
    folders_down = [folder_cloud_mask_planet, folder_label_mask_planet, folder_planet, folder_s2]
    if folder_cloud_mask_sentinel is not None:
        folders_down.append(folder_cloud_mask_sentinel)
    download_folders(folders_down,
                     force_download=force_download, trigger_download_if_not_exists=trigger_download_if_not_exists,
                     path_local=data_dir, path_bucket=PATH_BUCKET_SPACENET7)

    # These exists because we checked before in the assert
    paths_sentinel = sorted(glob(osp.join(data_dir, f'*/{folder_s2}/*')))
    paths_planet = sorted(glob(osp.join(data_dir, f'*/{folder_planet}/*')))

    ## TODO find folder for sentinel cloud masks.
    df = pd.concat([pd.DataFrame(dict(path=paths_sentinel, sat='sentinel')),
                    pd.DataFrame(dict(path=paths_planet, sat='planet'))], ignore_index=True)

    ## Scene names, filenames from paths.
    df['scene'] = df.path.map(lambda x: x.split(osp.sep)[4])
    df['basename'] = df['path'].map(lambda x: osp.basename(x))
    df[['year', 'month', 'day']] = 1

    ## Year, month, day from names (sentinel).
    ix = df.query('sat=="sentinel"').index
    df.loc[ix, ['year', 'month', 'day']]  = (df.loc[ix, 'basename']
                                             .str.extract(r'(\d{4})-(\d{2})-(\d{2})')
                                             .rename(columns={0:'year', 1:'month', 2:'day'})
                                             .astype(int))

    ## Year, month, day from names (planet; different format).
    ix = df.query('sat=="planet"').index
    df.loc[ix, ['year', 'month']] = (df.loc[ix, 'basename']
                                     .str.extract(r'monthly_(\d{4})_(\d{2})')
                                     .rename(columns={0:'year', 1:'month', 2:'day'})
                                     .astype(int))

    ## Datetimes from year, month, day.
    df['datetime'] = df.apply(lambda x: datetime(year=x['year'], month=x['month'], day=x['day']), axis=1)
    df.set_index(['sat', 'scene', 'datetime'], inplace=True)
    df.sort_index(inplace=True)

    ## Paths to cloud and labels masks.
    folder_cloud_mask_sentinel_str = "" if folder_cloud_mask_sentinel is None else folder_cloud_mask_sentinel

    def get_cloud_mask_path(path):
        return (re.sub(folder_planet, folder_cloud_mask_planet, path)
                .replace(folder_s2, folder_cloud_mask_sentinel_str)
                .replace('TCI', 'SCL'))

    def get_label_mask_path(path):
        return (re.sub(folder_planet, folder_label_mask_planet, path)
                .replace('.tif', '_Buildings.tif'))

    df['cloud_mask_path'] = df.path.map(get_cloud_mask_path)
    df['label_mask_path'] = df.path.map(get_label_mask_path)
    
    if folder_cloud_mask_sentinel is None:
        df.loc['sentinel', 'cloud_mask_path'] = None  # Sentinel cloud masks given as a TIFF band

    ## Split scenes into train / val / test
    scenes = df.index.get_level_values('scene')
    if random_split:
        partition = train_val_test_partition(scenes.unique(), split=train_val_test_split, seed=random_seed)
    else:
        partition = pd.read_json(split_json).to_dict()['split']
    df['split'] = scenes.map(partition)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Force download from the bucket")
    parser.add_argument(
        "--problem",
        choices=["wotus", "spacenet", "spacenet6"],
        action='store', type=str, default="spacenet",
        help='Problem to download the data from bucket',
    )
    args = parser.parse_args()
    
    if args.problem == "spacenet":
        spacenet7_index(force_download=True)
    elif args.problem == "wotus":
        wotus_index(force_download=True)
    else:
        raise NotImplementedError(f"Problem {args.problem} not implemented yet")
