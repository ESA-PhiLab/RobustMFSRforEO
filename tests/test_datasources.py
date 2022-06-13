import os.path as osp
import pytest
import rasterio
import numpy as np

from src.datasources import spacenet7_index, wotus_index, glob_gsutil
from src import datasources

dataframes = [
    ## Add here all dataset indices.
    {"dataframe": spacenet7_index(),
     "channels_s2" : 13,
     "channels_planet" : 4,
     "folder_planet" : "images",
     "folder_s2": "S2L2A",
     "bucket_path": datasources.PATH_BUCKET_SPACENET7,
     "n_items": 2262},
]

# TODO add to dataframes list when Wotus dataset/loader is prepared
test_wotus = True
if test_wotus:
    dataframes.append({"dataframe": wotus_index(),
                       "channels_s2" : 14, # It has B10 band which is not in S2L2A products
                       "channels_planet" : 4,
                       "folder_planet" : "images",
                       "folder_s2": "S2",
                       "bucket_path": datasources.PATH_BUCKET_WOTUS,
                       "n_items": 225})


@pytest.mark.parametrize("df_info", dataframes)
def test_df_train_val_test_split(df_info):
    df = df_info["dataframe"]
    train = set(df.query(f'split=="train"').index.get_level_values('scene').unique())
    val = set(df.query(f'split=="val"').index.get_level_values('scene').unique())
    test = set(df.query(f'split=="test"').index.get_level_values('scene').unique())
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(train) + len(val) + len(test) == len(train | val | test)


@pytest.mark.parametrize("df_info", dataframes)
def test_df_multiindex(df_info):
    df = df_info["dataframe"]
    assert df.index.get_level_values('sat').size == df_info["n_items"], \
    f"Found {df.index.get_level_values('sat').size} expected {df_info['n_items']}" 


@pytest.mark.parametrize("df_info", dataframes)
def test_df_images_exist(df_info):
    df = df_info["dataframe"]
    assert all(osp.exists(p) for p in df['path'])

    
@pytest.mark.parametrize("df_info", dataframes)
def test_expected_number_of_channels(df_info):
    df = df_info["dataframe"]
    for query_sat, n_channels in zip(["sentinel", "planet"],
                                        [df_info["channels_s2"], df_info["channels_planet"]]):
        assert all(rasterio.open(p).count == n_channels for p in df.loc[query_sat]['path'])


@pytest.mark.parametrize("df_info", dataframes)
def test_expected_geographic_info_planet_sentinel(df_info):
    df = df_info["dataframe"]

    df_planet = df.loc["planet"]
    df_sentinel = df.loc["sentinel"]

    assert df_sentinel.shape[0] > 0, "There are not Sentinel images in the dataframe" 

    n_tests_run = 0
    for tuple_s2 in df_sentinel.itertuples():

        scene_id = tuple_s2.Index[0]
        df_planet_scene = df_planet.loc[scene_id]

        if df_planet_scene.shape[0] > 1:
            # TODO testing only 2020 for SpaceNet-7
            date_planet = "2020-01-01"
            assert date_planet in df_planet_scene.index, f"{tuple_s2.path} not found in Planet (there is a S2 image with no correponding Planet)"
        else:
            date_planet = df_planet_scene.index[0]

        planet_register = df_planet_scene.loc[date_planet]
        with rasterio.open(planet_register.path) as src:
            transform_planet =  src.transform
            crs_planet = src.crs
            shape_planet = src.shape
        with rasterio.open(tuple_s2.path) as src2:
            transform_s2 = src2.transform
            crs_s2 = src2.crs
            shape_s2 = src2.shape

        assert equal_float(abs(transform_s2.a), 10.) and equal_float(abs(transform_s2.e), 10.), f"{tuple_s2.path} does not have 10m GSD"
        assert equal_float(transform_planet.f, transform_s2.f, 1e-3) and equal_float(transform_planet.c, transform_s2.c, 1e-3), f"Different transform: {planet_register.path} and {tuple_s2.path}"
        assert crs_planet == crs_s2, f"Different CRS: {planet_register.path} and {tuple_s2.path}"

        expected_shape = np.ceil(np.array(shape_planet)*abs(transform_planet.a)/abs(transform_s2.a)).astype(int).tolist()

        assert tuple(expected_shape) == shape_s2, f"Unexpected shape {planet_register.path} and {tuple_s2.path}"

def equal_float(a, b, eps=1e-4):
    return abs(a-b) < eps

@pytest.mark.parametrize("df_info", dataframes)
def test_files_synced_with_bucket(df_info):
    df = df_info["dataframe"]
    for query_sat, remote_folder in zip(["sentinel","planet"],
                                        [df_info["folder_s2"],df_info["folder_planet"]]):
        sorted_paths_local = sorted(df.loc[query_sat]['path'].map(lambda x: "/".join(x.split("/")[-3:])).tolist())
        sorted_paths_remote = sorted(["/".join(path_remote.split("/")[-3:]) for path_remote in datasources.glob_gsutil(osp.join(df_info["bucket_path"],"*",remote_folder,"*"))])
        
        assert sorted_paths_local == sorted_paths_remote, f"Folder {remote_folder} expected: {len(sorted_paths_remote)} found: {len(sorted_paths_local)}"


@pytest.mark.parametrize("df_info", dataframes)
def test_df_planet_cloud_masks_exist(df_info):
    df = df_info["dataframe"]
    # Wotus data does not have cloud masks
    if "cloud_mask_path" in df.columns:
        assert df.loc['planet']['cloud_mask_path'].map(osp.exists).all()
    


# @pytest.mark.parametrize("df", dataframes)
# def test_df_label_masks_exist(df):
#     assert all([osp.exists(p) for p in df['label_mask_path']])
