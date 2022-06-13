from src.datamodules import SN7DataModule
from src.datasources import MNT_SPACENET7
import torch
import pytorch_lightning as pl
import numpy as np
import os
import rasterio
import torch.nn.functional
import tempfile
import subprocess
import re
import json
from tqdm import tqdm
from typing import Optional, List, Dict


PATH_SAVE_PREDS = f"gs://fdl_srhallucinate/spacenet/predictions/"


@torch.no_grad()
def predictions_to_bucket(model:pl.LightningModule, name_prediction_folder:str,
                          data_module:SN7DataModule=None, verbose:bool=False)->List[Dict]:
    f"""
    Make predictions with given model and save the in {PATH_SAVE_PREDS}/name_prediction_folder folder. 
    It expects a model with  attribute scale:bool that controls wether to do an additional scaling to the network. 
    This attribute will be set to False and the additional scaling will be held manually for each highres-lowres pair 

    Args:
        model: pl.LightningModule with attributes eval(), scale and device
        name_prediction_folder: preds in the bucket will be saved in {PATH_SAVE_PREDS}/name_prediction_folder folder
        data_module: SN7DataModule. The function will call setup() will if attributes only_whole_scenes is False or batch_size!=1.
        verbose: if True prints lowres,highres,superres shapes in each iteration

    Returns:
        None
    """

    if not data_module.only_whole_scenes or (data_module.batch_size != 1):
        data_module.only_whole_scenes = True
        data_module.batch_size = 1
        data_module.setup()


    model.eval()
    folder_preds = os.path.join(PATH_SAVE_PREDS, name_prediction_folder)
    print(f"Copying predictions to: {folder_preds}")

    preds_toupload_gee = []
    for dl, dl_name in zip([data_module.val_dataloader(), data_module.test_dataloader(), data_module.train_dataloader()],
                          ["val", "test", "train"]):
        for idx, batch in enumerate(dl):
            scene_id = batch["highres"]["scene"][0]
            print(f"Dataloader {dl_name} ({idx}/{len(dl.dataset)}) making predictions for scene {scene_id}")
            lowres, highres = batch['lowres'], batch["highres"]

            assert lowres['images'].shape[0] == 1, "Expected batch size 1. This is needed otherwise it pads the inputs"

            # Send all tensors in lowres to model.device
            if model.device != torch.device("cpu"):
                for key, value in lowres.items():
                    if isinstance(value, torch.Tensor):
                        lowres[key] = value.to(model.device)

            # This is solved by changing the internal additional scaling parameter self.net.scale
            # _, _, _, H, W = lowres['images'].shape
            # if H > W:
            #     S = W
            # elif W > H:
            #     S = H
            # else:
            #     S = H
            #
            # if S % 2 == 1:
            #     S -= 1

            superres_tensor = model.prediction_step(lowres)
            if verbose:
                print(f"\t Input shape: {lowres['images'].shape} output shape: {superres_tensor.shape} highres shape: {highres['images'].shape}")

            assert superres_tensor.dim() == 4, f"Unexpected number of dimensions expected (B,C, H, W) found {superres_tensor.dim()}"
            assert tuple(superres_tensor.shape[:2]) == (1, highres['images'].shape[2]), \
                f"Unexpected shape of the superres tensor expected: {(1, highres.shape[2])} found {tuple(superres_tensor.shape[:2])} hint: use batch_size=1"

            if superres_tensor.shape[2:] != highres['images'].shape[3:]:
                superres_tensor = torch.nn.functional.interpolate(superres_tensor,
                                                                  size=highres['images'].shape[3:],
                                                                  mode="bicubic", align_corners=False)
                superres_tensor = torch.clamp(superres_tensor, 0, 1)

            assert superres_tensor.shape[-2:] == highres["images"].shape[-2:], \
                f"Different shapes after superpres model: {superres_tensor.shape[-2:]} {highres['images'].shape[-2:]}"

            superres = np.round(superres_tensor.numpy() * 255).astype(np.uint8)[0] # (C, H, W)

            # Take geographic info to save the prediction
            # TODO take this path from the guts of the dataloader  (dl.dataset.df.loc[sat,scene,??]
            highres_path = os.path.join(MNT_SPACENET7, scene_id, "images", f"global_monthly_2020_01_mosaic_{scene_id}.tif")

            basename_highres_path = os.path.basename(highres_path)
            with rasterio.open(highres_path) as raster:
                profile = {
                    "dtype": rasterio.uint8,
                    "crs": raster.crs,
                    "nodata": 0.,
                    "compress": "lzw",
                    "RESAMPLING": "CUBICSPLINE",  # for pyramids
                    "transform": raster.transform,
                }
                raster_shape = raster.shape

            assert raster_shape == tuple(highres["images"].shape[-2:]), \
                f"Unexpected shapes raster and loaded image scene: {scene_id} with shape {raster_shape} {highres['images'].shape[-2:]} hint: use only_whole_scenes=1"

            path_2_save_sr = os.path.join(folder_preds, basename_highres_path)
            save_cog_copy_gs(superres, path_2_save_sr, profile)
            preds_toupload_gee.append({"scene_id": scene_id, "split": dl_name, "path": path_2_save_sr})

    return preds_toupload_gee


def save_cog(out_np, path_tiff_save, profile, tags=None):
    """
    saves `out_np` np array as a COG GeoTIFF in path_tiff_save. profile is a dict with the geospatial info to be saved
    with the TiFF.
    :param out_np: 3D numpy array to save in CHW format
    :param path_tiff_save:
    :param profile: dict with profile to write geospatial info of the dataset: (crs, transform)
    :param tags: extra info to save as tags
    """
    # Set count, height, width
    for idx, c in enumerate(["count", "height", "width"]):
        if c in profile:
            assert profile[c] == out_np.shape[idx], f"Unexpected shape: {profile[c]} {out_np.shape}"
        else:
            profile[c] = out_np.shape[idx]
    for field in ["crs","transform"]:
        assert field in profile, f"{field} not in profile. it will not write cog without geo information"
    profile["BIGTIFF"] = "IF_SAFER"
    with rasterio.Env() as env:
        assert  "COG" in env.drivers(), "Expected COG driver"
    profile["driver"] = "COG"
    with rasterio.open(path_tiff_save, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
    return path_tiff_save


def save_cog_copy_gs(destination_array, path_save_tiff_gs, profile, tags=None):
    """ Save in tmp file and copy to gs bucket """
    fileobj = tempfile.NamedTemporaryFile(dir=".", suffix=".tif", delete=True)
    named_tempfile = fileobj.name
    fileobj.close()
    save_cog(destination_array, named_tempfile, profile, tags=tags)
    subprocess.run(["gsutil", "cp", named_tempfile, path_save_tiff_gs])
    if os.path.exists(named_tempfile):
        os.remove(named_tempfile)
    return named_tempfile


def create_image_collection_gee(preds_toupload_gee:List[Dict], name_prediction_folder:str, gee_user:str) -> None:
    """ This function requires that name_prediction_folder exists in the gee_user assets """
    
    request_template = {
        'type': 'IMAGE',
        'gcs_location': {
            'uris': ['gs://ee-docs-demos/COG_demo.tif']
        },
        'properties': {
            'split': 'train'
        },
        'startTime': '2016-01-01T00:00:00.000000000Z',
        'endTime': '2016-12-31T15:01:23.000000000Z',
    }

    asset_collection_name = f"users/{gee_user}/{name_prediction_folder}"
    url = 'https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/assets?assetId={}'

    print("Creating Image Collection in the GEE")
    session = gee_rest_api_session()
    for f2upload in tqdm(preds_toupload_gee, desc="Creating Image Collection in GEE..."):
        fpf = f2upload["path"]
        scene_id = f2upload["scene_id"]
        basename_pred = os.path.splitext(os.path.basename(fpf))[0]
        year, month = re.match("global_monthly_(\d{4})_(\d{2})_mosaic_", basename_pred).groups()
        start_time = f'{year}-{month}-01T00:00:00.000000000Z'
        end_time = f'{year}-{month}-31T23:59:00.000000000Z'
        request_copy = request_template.copy()
        request_copy['gcs_location']['uris'] = [fpf]
        request_copy['startTime'] = start_time
        request_copy['endTime'] = end_time
        request_copy["properties"] = {"scene_id": scene_id, "split": f2upload["split"]}

        asset_name = f"{scene_id}-{year}-{month}"
        asset_id = f"{asset_collection_name}/{asset_name}"

        response = session.post(
            url=url.format(asset_id),
            data=json.dumps(request_copy))

        assert response.status_code == 200, f'{json.loads(response.content)}'


def gee_rest_api_session():
    from ee import oauth
    from google_auth_oauthlib.flow import Flow
    import json

    # Build the `client_secrets.json` file by borrowing the
    # Earth Engine python authenticator.
    client_secrets = {
        'web': {
            'client_id': oauth.CLIENT_ID,
            'client_secret': oauth.CLIENT_SECRET,
            'redirect_uris': [oauth.REDIRECT_URI],
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://accounts.google.com/o/oauth2/token'
        }
    }

    # Write to a json file.
    client_secrets_file = 'client_secrets.json'
    with open(client_secrets_file, 'w') as f:
        json.dump(client_secrets, f, indent=2)

    # Start the flow using the client_secrets.json file.
    flow = Flow.from_client_secrets_file(client_secrets_file,
                                         scopes=oauth.SCOPES,
                                         redirect_uri=oauth.REDIRECT_URI)

    # Get the authorization URL from the flow.
    auth_url, _ = flow.authorization_url(prompt='consent')

    # Print instructions to go to the authorization URL.
    oauth._display_auth_instructions_with_print(auth_url)
    print('\n')

    # The user will get an authorization code.
    # This code is used to get the access token.
    code = input('Enter the authorization code: \n')
    flow.fetch_token(code=code)

    # Get an authorized session from the flow.
    return flow.authorized_session()