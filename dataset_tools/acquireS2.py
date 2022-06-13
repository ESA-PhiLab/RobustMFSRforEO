import os
import sys
import subprocess
import rasterio
from dataset_tools import S2_image
from dataset_tools.utils_gis import save_cog, polygon_bounds, transform_bounds_res
import numpy as np
from datetime import datetime, timedelta, timezone
import shapely.wkt
from collections import OrderedDict
import argparse
import re
import shutil
from tqdm import tqdm
import tempfile


def exists_bucket_or_disk(path):
    """ Function that checks if a path file exists to be used as drop-in replace for os.path.exists """
    if path.startswith("gs://"):
        from google.cloud import storage
        splitted_path = path.replace("gs://","").split("/")
        bucket_name = splitted_path[0]
        blob_name = "/".join(splitted_path[1:])
        buck = storage.Client().get_bucket(bucket_name)
        return buck.get_blob(blob_name) is not None
    return os.path.exists(path)


def get_api():
    from sentinelsat.sentinel import SentinelAPI
    return SentinelAPI('gonzmg88', '7qdV9CR7iQwL_hK', 'https://scihub.copernicus.eu/apihub')


def download(planet_obj, api, path_products, products_triggered=None, producttype="S2MSI1C"):
    """
    Downloads all images that overlap planet_path raster between -+daysaddsubtract

    Operations are idempotent (does not download if file exists)

    :param planet_obj:
    :param api: sentinelsat api object
    :param path_products: loc to save the downloaded data gs://fdl_superres ...
    :param products_triggered: Dict to check if this data has already been triggered.
    :param producttype: Sentinel-2 product type, with atmospheric correction (S2MSI2A) or without it (S2MSI1C)

    :return:dict of products triggered that couldn't be downloaded (because they are on the Long term archive LTE)
    """
    if products_triggered is None:
        products_triggered = OrderedDict()

    from sentinelsat import sentinel

    pol_aoi = planet_obj.polygon()

    if not path_products.startswith("gs://"):
        os.makedirs(path_products, exist_ok=True)

    products = api.query(area=str(pol_aoi),
                         date=(planet_obj.datetime_start.strftime("%Y%m%d"), planet_obj.datetime_end.strftime("%Y%m%d")),
                         platformname='Sentinel-2',
                         producttype=producttype,
                         cloudcoverpercentage=(0, 100))

    print(f"File {planet_obj} Date: {planet_obj.datetime.strftime('%Y-%m-%d')} found {len(products)} S2 products")

    ndown = 0
    for k, v in products.items():
        g = shapely.wkt.loads(v["footprint"])

        # Check overlap
        min_area = min(g.area, pol_aoi.area)
        inter = pol_aoi.intersection(g)
        per_ov = inter.area / min_area
        if per_ov <= .05:
            continue

        ndown += 1
        if k in products_triggered:
            print(f"Product {k} already triggered to download")
            continue
        product_filename = os.path.join(path_products, v["title"] + ".zip")
        if exists_bucket_or_disk(product_filename):
            print(f"Product {k} already downloaded in folder")
            continue

        path_download_temp = "."
        try:
            product = api.download(k, path_download_temp)
            if not product["Online"]:
                products_triggered[k] = product
                continue
        except sentinel.SentinelAPILTAError as e:
            print(f"Error in API this product won't be triggered. \nError message: {e.msg}")
            continue

        file_download_temp = os.path.join(path_download_temp, os.path.basename(product_filename))

        if not os.path.exists(file_download_temp):
            print(f"Error in download file {product_filename}")
            continue
               
        # Copy zip to bucket
        subprocess.run(["gsutil","-m","cp",file_download_temp, product_filename])
               
        # Unzip file
        unzip_sentinel2(product_filename, file_download_temp)

        # Remove zip
        if os.path.exists(file_download_temp):
            os.remove(file_download_temp)

    print(f"\tTriggered products with overlap: {ndown}")

    return products_triggered

# def wait_products(products_triggered, path_products):
#     if products_triggered is None:
#         return
#
#     while len(products_triggered) > 0:
#         for k, v in products_triggered.items():
#             status = api.get_product_odata(k)
#             if status["Online"]:
#                 product_filename = os.path.join(path_products, status["title"] + ".zip")
#                 if os.path.exists(product_filename):
#                     del products_triggered[k]
#                     continue
#
#                 path_download_temp = "."
#                 api.download(k, path_download_temp)
#
#                 file_download_temp = os.path.join(path_download_temp, os.path.basename(product_filename))
#
#                 if not os.path.exists(file_download_temp):
#                     print(f"Error in download file {product_filename}")
#                     continue
#
#                 # Copy zip to bucket
#                 subprocess.run(["gsutil", "-m", "cp", file_download_temp, product_filename.replace("/mnt","gs:/")])
#
#                 # Unzip file
#                 unzip_sentinel2(product_filename, file_download_temp)
#
#                 # Remove zip
#                 if os.path.exists(file_download_temp):
#                     os.remove(file_download_temp)
#
#         time.sleep(5*60)
#         print(f"Number of products waiting for downloading {len(products_triggered)}")


def unzip_sentinel2(s2zip_file, file_temp_zip=None):
    """ Unzip s2 file. check if file exists before unzipping. Copy the file to bucket

    :param s2zip_file Could be gs://
    :param file_temp_zip. Must be local file

    """
    file_path_noext, _ = os.path.splitext(s2zip_file)
    s2_path_unzip = file_path_noext + ".SAFE"
    if exists_bucket_or_disk(s2_path_unzip):
        return
    folder_dest = os.path.dirname(s2zip_file)
    
    file_zip = s2zip_file if file_temp_zip is None else file_temp_zip
    subprocess.run(["unzip",file_zip,  "-d", "."])
    
    dst_tmp = os.path.join(".", os.path.basename(s2_path_unzip))

    if not os.path.exists(dst_tmp):
        print(f"{s2zip_file} was not properly unziped, {dst_tmp} does not exists.")
        return

    subprocess.run(["gsutil","-m","mv", "-r", dst_tmp, folder_dest])

    # gsutil mv removes the files but not the directories
    shutil.rmtree(dst_tmp)


def sentinel2_to_cloud_mask_preprocess(x):
    """
    takes x in the format of the tif file and rescales it to the format that s2 cloudless expects.
    """
    # last channel is a 'quality assesment' channel rather than a sensor input
    # s2 cloudless also expects channels last and to be scaled to 0-1

    return x.transpose(1, 2, 0)[None, ...] / 10000.


def compute_cloud_mask(x):
    from s2cloudless import S2PixelCloudDetector
    z = sentinel2_to_cloud_mask_preprocess(x)
    cloud_detector = S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=True
    )
    return cloud_detector.get_cloud_probability_maps(z)


def reproject_mosaic_s2(s2objs, planet_obj, res_s2=10., producttype=None):
    """ Reproject and mosaic s2 images to the same CRS and bounds that planet (but using {res_s2} resolution) """
    assert producttype in {"S2MSI2A", "S2MSI1C"}, f"Unrecognized product type {producttype}"

    s2datestr = s2objs[0].datetime.strftime("%Y-%m-%d")
    assert len(s2objs) > 0, "s2obj empty list"
    assert all(s2obj.datetime.strftime("%Y-%m-%d") == s2datestr for s2obj in s2objs), \
        f"Expected all products from the same date found {[s2obj.datetime.strftime('%Y-%m-%d') == s2datestr for s2obj in s2objs]}"

    # # planet_path is /mnt/fdl_superres/{problem}/train/*/images/{globtiff}*.tif
    scene_folder = os.path.dirname(os.path.dirname(planet_obj.path)) # Remote folder in many cases

    # Compute filenames and check if exists (to make operations idempotent)
    path_save_s2 = os.path.join(scene_folder, PRODUCTTYPE_FOLDER["scene"][producttype], s2datestr+".tif")
    path_save_s2_valid = os.path.join(scene_folder, PRODUCTTYPE_FOLDER["valid"][producttype], s2datestr+".tif")

    # Sort s2objs by area overlap
    planet_pol = planet_obj.polygon()
    s2objs = sorted(s2objs, key= lambda s2: s2.polygon().intersection(planet_pol).area/planet_pol.area,
                    reverse=True)

    if exists_bucket_or_disk(path_save_s2) and exists_bucket_or_disk(path_save_s2_valid):
        # Check the mosaic contains all the products
        with rasterio.open(path_save_s2) as src_s2:
            tags = src_s2.tags()

        # Return if file does not have invalids or we've computed the composite with the same S2 files
        if ("s2_files" in tags) and ("frac_invalids" in tags) and (float(tags["frac_invalids"]) < (1/1000.)):
            return
        if ("s2_files" in tags) and ("frac_invalids" in tags) and (eval(tags["s2_files"]) == [s2obj.folder for s2obj in s2objs]):
            return

    # Starting process
    print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} Reproject and mosaic {','.join([s2obj.folder for s2obj in s2objs])} for Planet image {planet_obj.path}. Output: {path_save_s2_valid}")

    # Obtain info of Planet image for reproject
    dst_crs = planet_obj.crs
    planet_bounds = planet_obj.bounds
    src_planet_transform = planet_obj.transform

    assert (src_planet_transform.b < 1e-6) and (src_planet_transform.d < 1e-6), f"Non squared transform Reference Tiff: {src_planet_transform}"

    destination_array, current_clouds = S2_image.mosaic_s2(s2objs, planet_bounds,
                                                           dst_crs, res_s2=res_s2)

    # add cloud mask
    # S2MSI2A include the SCL band (see S2_image.S2L2AImage class)
    if s2objs[0].producttype == "MSIL1C":
        current_clouds = compute_cloud_mask(destination_array)
        # save valid mask as COG GeoTIFF
        valid_mask = np.any((destination_array > 0) & (destination_array < (2**16 - 1)),
                            axis=0, keepdims=True).astype(np.uint8) * 2
        valid_mask[(current_clouds>.5) & (valid_mask == 2)] = 3  # {0: invalid, 1: shouldnothappen, 2: valid, 3: clouds}
        current_clouds+=1
        current_clouds[valid_mask == 0] = 0. # 0: invalid, (>=1 & <=1.5): clear (>1.5): cloud
    else:
        valid_mask = current_clouds.astype(np.uint8)
        valid_mask = valid_mask[None]
        current_clouds = current_clouds[None]

    destination_array = np.concatenate([destination_array,
                                        current_clouds.astype(np.float32)], axis=0)

    profile = {
        "dtype": rasterio.float32,
        "crs": dst_crs,
        "nodata": 0.,
        "compress": "lzw",
        "RESAMPLING": "CUBICSPLINE", # for pyramids
        "transform": transform_bounds_res(planet_bounds, (res_s2, res_s2)),
    }

    # add the dates as a tag
    tags = {"s2_files": [s2obj.folder for s2obj in s2objs],
            "frac_invalids": float(1-(np.sum(valid_mask > 1)/np.prod(valid_mask.shape)))}

    # save bands and cloud mask as COG GeoTIFF
    save_cog_copy_gs(destination_array, path_save_s2, profile, tags)

    profile_invalid = profile.copy()
    profile_invalid.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0,
                            "RESAMPLING": "NEAREST"})  # for pyramids
    save_cog_copy_gs(valid_mask, path_save_s2_valid, profile_invalid, tags)


def save_cog_copy_gs(destination_array, path_save_tiff_gs, profile, tags):
    """ Save in tmp file and copy to gs bucket """
    fileobj = tempfile.NamedTemporaryFile(dir=".", suffix=".tif", delete=True)
    named_tempfile = fileobj.name
    fileobj.close()
    save_cog(destination_array, named_tempfile, profile, tags=tags)
    subprocess.run(["gsutil", "cp", named_tempfile, path_save_tiff_gs])
    if os.path.exists(named_tempfile):
        os.remove(named_tempfile)
    return named_tempfile


def convert_cog_referencetiff(pobjs):
    """ Convert to COG reference tiff images. Requires gdal_translate and GDAL Version >=3.1 """
    for _i, pobj in enumerate(pobjs):
        pl_path = pobj.path
        out_path = pl_path.replace("/images/", "/imagesCOG/")
        if exists_bucket_or_disk(out_path):
            continue

        print(f"({_i} / {len(pobjs)}) Converting {pl_path} to COG")
        fileobj = tempfile.NamedTemporaryFile(dir=".", suffix=".tif", delete=True)
        named_tempfile = fileobj.name
        fileobj.close()

        subprocess.run(["rio", "convert", pl_path, named_tempfile, "--driver", "COG",
                        "--co", "BIGTIFF=IF_SAFER", "--co", "RESAMPLING=CUBICSPLINE"])

        if os.path.exists(named_tempfile):
            subprocess.run(["gsutil", "-m", "cp", named_tempfile, out_path])
            os.remove(named_tempfile)


def glob_gsutil(globloc, isdir=False):
    """" Globbing in the mounted location hangs """

    print(f"Globbing {globloc}")
    if isdir:
        commands = ["gsutil", "ls", "-d", globloc]
    else:
        commands = ["gsutil", "ls", globloc]

    glob_result = subprocess.check_output(commands)
    return [g for g in glob_result.decode("UTF-8").split('\n') if g != ""]


def copy_and_reprojects2images(s2objs, pobjs, producttype):
    """ This is basically an inner join of s2prods_dict and pobjs by date and location """

    # mosaic and reproject all products
    print("Start Reproject Products:")
    for _i, pobj in enumerate(pobjs):
        print(
            f"{_i}/{len(pobjs)} {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} Checking ref img {pobj}")
        s2overlap = query_s2_files_spatiotemporal_overlap(pobj, s2objs)
        s2group = s2_products_grouped(s2overlap)
        for date_str, list_prods in s2group.items():
            reproject_mosaic_s2(list_prods, pobj, producttype=producttype)


def s2_products_grouped(s2objs):
    """ Creates S2Image objects from the unzipped file locations and groups them by date """
    prods_date = OrderedDict()
    for s2 in s2objs:
        date_str = s2.datetime.strftime("%Y-%m-%d")
        if date_str not in prods_date:
            prods_date[date_str] = []
        prods_date[date_str].append(s2)

    return prods_date


def overlap_pols(pol1, pol2, area_overlap=0.05):
    intersection_bounds = pol1.intersection(pol2)
    return (not intersection_bounds.is_empty) and ((intersection_bounds.area / pol1.area) >= area_overlap)


def query_s2_files_spatiotemporal_overlap(ref_tif_obj, list_s2_objs, area_overlap=.05):
    """ Return objs from list_s2_objs that overlap  spatio temporally ref_tif_obj """
    if isinstance(ref_tif_obj,str):
        ref_tif_obj = ReferenceTiff(ref_tif_obj)

    ret_objs = []
    for s2obj in list_s2_objs:
        if isinstance(s2obj,str):
            s2obj = S2_image.s2loader(s2obj)

        if (s2obj.datetime < ref_tif_obj.datetime_start) | (s2obj.datetime > ref_tif_obj.datetime_end):
            continue

        if overlap_pols(ref_tif_obj.polygon(), s2obj.polygon(), area_overlap=area_overlap):
            ret_objs.append(s2obj)

    return  ret_objs


PRODUCTTYPE_FOLDER = {
    "raw" : {
        "S2MSI1C" : "S2raw",
        "S2MSI2A" : "S2L2Araw",
    },
    "scene": {
        "S2MSI1C": "S2",
        "S2MSI2A": "S2L2A",
    },
    "valid": {
        "S2MSI1C": "valid_mask_S2",
        "S2MSI2A": "valid_mask_S2L2A",
    }
}

DEFAULT_DAYS_ADD_SUBTRACT = 33
FORMATS = ["(\d{4})-(\d{2})-(\d{2})", "global_monthly_(\d{4})_(\d{2})_mosaic_"]


class ReferenceTiff:
    """ Dummy class to avoid opening the remote geotiff """
    def __init__(self, ref_path, daysaddsubtract=DEFAULT_DAYS_ADD_SUBTRACT):
        with rasterio.open(ref_path) as src:
            self.meta = src.meta
            self.meta["shape"] = src.shape
            self.meta["res"] = src.res
            self.pol_bounds_epsg4326 = polygon_bounds(src)
            self.meta["bounds"] = src.bounds

        self.path = ref_path  # assume remote

        planet_basename_noext = os.path.splitext(os.path.basename(ref_path))[0]

        # Load date_scene
        date_scene = None
        for f in FORMATS:
            matches = re.match(f, planet_basename_noext)
            if matches is not None:
                groups = matches.groups()
                if len(groups) == 2:
                    date_scene = datetime.strptime(f"{groups[0]}-{groups[1]}-01", "%Y-%m-%d")
                else:
                    date_scene = datetime.strptime(f"{groups[0]}-{groups[1]}-{groups[2]}", "%Y-%m-%d")
                break

        assert date_scene is not None, f"Could not find date in file: {planet_basename_noext}"

        self.datetime = date_scene.replace(tzinfo=timezone.utc)

        self.datetime_start = self.datetime - timedelta(days=daysaddsubtract)
        self.datetime_end = self.datetime + timedelta(days=daysaddsubtract)

    def __str__(self):
        return self.path

    def polygon(self):
        return self.pol_bounds_epsg4326

    @property
    def shape(self):
        return self.meta["shape"]

    @property
    def transform(self):
        return self.meta["transform"]

    @property
    def bounds(self):
        return self.meta["bounds"]

    @property
    def crs(self):
        return self.meta["crs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        help='Trigger download from scihub.copernicus.eu',
        action='store_true'
    )
    parser.add_argument(
        "--problem",
        choices=["wotus", "spacenet", "spacenet6"],
        action='store', type=str,
        help='Problem to download the data',
    )
    parser.add_argument(
        "--producttype",
        choices=["S2MSI1C", "S2MSI2A"],
        action='store', type=str,
        help='Prodct type of the data to download',
    )
    parser.add_argument(
        "--daysaddsubtract",
        action='store', type=int, default=DEFAULT_DAYS_ADD_SUBTRACT,
        help='Which planet tiffs to glob: (2020 for SpaceNet)',
    )
    parser.add_argument(
        "--globtiff",
        action='store', type=str, default="",
        help='Which planet tiffs to glob: (2020 for SpaceNet)',
    )
    args = parser.parse_args()

    problem = args.problem
    ptype = args.producttype

    # SpaceNet use the same raw folder (Rotterdam image is in both spacenet6 and spacenet7
    problem_s2_raw = "spacenet" if problem.startswith("spacenet") else problem
    path_ds_prods_raw = f"gs://fdl_srhallucinate/{problem_s2_raw}/{PRODUCTTYPE_FOLDER['raw'][ptype]}/"

    # Patch to only download S2 images for specific PlanetScope data (i.e. 2020)
    globtiff = f"*{args.globtiff}" if args.globtiff != "" else ""
    planet_objs = [ReferenceTiff(p,daysaddsubtract=args.daysaddsubtract) for p in sorted(glob_gsutil(f"gs://fdl_srhallucinate/{problem}/train/*/images/{globtiff}*.tif"))]
    print(f"Found {len(planet_objs)} Planet paths")

    # Download products
    if args.download:
        api = get_api()
        products_trig = None
        for p_path in planet_objs:
            print(f"Trigger download for Planet file {p_path}")
            products_trig = download(p_path, api, path_ds_prods_raw,
                                     products_triggered=products_trig, producttype=ptype)
        if len(products_trig) > 0:
            print(f"There are {len(products_trig)} pending to be downloaded")

    # Convert to COG to show in GEE
    if problem.startswith("spacenet"):
        convert_cog_referencetiff(planet_objs)

    # Group S2 products by date
    print("Creating S2 raw products")
    constructor = S2_image.S2ImageL2A if ptype == "S2MSI2A" else S2_image.S2ImageL1C
    products_date = [constructor(s) for s in tqdm(sorted(glob_gsutil(os.path.join(path_ds_prods_raw, "*.SAFE"), isdir=True)))]

    print("Start copy and reproject")
    copy_and_reprojects2images(products_date, planet_objs, producttype=ptype)
