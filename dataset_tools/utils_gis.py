import rasterio
import rasterio.rio.overview
import rasterio.shutil as rasterio_shutil
from shapely.geometry import Polygon
import os
import tempfile
import rasterio.warp
from rasterio import Affine


def add_overviews(rst_out, tile_size, verbose=False):
    """ Add overviews to be a cog and be displayed nicely in GIS software """

    overview_level = rasterio.rio.overview.get_maximum_overview_level(*rst_out.shape, tile_size)
    overviews = [2 ** j for j in range(1, overview_level + 1)]

    if verbose:
        print(f"Adding pyramid overviews to raster {overviews}")

    # Copied from https://github.com/cogeotiff/rio-cogeo/blob/master/rio_cogeo/cogeo.py#L274
    rst_out.build_overviews(overviews, rasterio.warp.Resampling.average)
    rst_out.update_tags(ns='rio_overview', resampling='nearest')
    tags = rst_out.tags()
    tags.update(OVR_RESAMPLING_ALG="NEAREST")
    rst_out.update_tags(**tags)
    rst_out._set_all_scales([rst_out.scales[b - 1] for b in rst_out.indexes])
    rst_out._set_all_offsets([rst_out.offsets[b - 1] for b in rst_out.indexes])


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
        cog_driver =  "COG" in env.drivers()


    if cog_driver:
        profile["driver"] = "COG"
        with rasterio.open(path_tiff_save, "w", **profile) as rst_out:
            if tags is not None:
                rst_out.update_tags(**tags)
            rst_out.write(out_np)
        return path_tiff_save

    print("COG driver not available. Generate COG manually with GTiff driver")
    # If COG driver is not available (GDAL < 3.1) we go to copying the file using GTiff driver
    # Set blockysize, blockxsize
    for idx, b in enumerate(["blockysize","blockxsize"]):
        if b in profile:
            assert profile[b] <= 512, f"{b} is {profile[b]} must be <=512 to be displayed in GEE "
        else:
            profile[b] = min(512, out_np.shape[idx+1])

    if (out_np.shape[1] >= 512) or (out_np.shape[2] >= 512):
        profile["tiled"] = True
    
    profile["driver"] = "GTiff"
    dir_tiff_save = os.path.dirname(path_tiff_save)
    fileobj = tempfile.NamedTemporaryFile(dir=dir_tiff_save, suffix=".tif", delete=True)
    named_tempfile = fileobj.name
    fileobj.close()

    with rasterio.open(named_tempfile, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
        add_overviews(rst_out, tile_size=profile["blockysize"])
        print("Copying temp file")
        rasterio_shutil.copy(rst_out, path_tiff_save, copy_src_overviews=True, tiled=True, blockxsize=profile["blockxsize"],
                             blockysize=profile["blockysize"],
                             driver="GTiff")

    rasterio_shutil.delete(named_tempfile)

    return path_tiff_save


def polygon_bounds(src_rasterio, dst_crs={'init': 'epsg:4326'}):
    """
    Obtain the polygon with the bounds of the raster in the dst_crs coords

    :param src_rasterio:
    :param dst_crs:
    :return:
    """

    bbox = src_rasterio.bounds
    bbox_lnglat = rasterio.warp.transform_bounds(src_rasterio.crs,
                                                 dst_crs,
                                                 *bbox)
    return Polygon(generate_polygon(bbox_lnglat))


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]

def transform_bounds_res(bbox_read, resolution_dst_crs):
    """ Compute affine transform for a given bounding box and resolution. bbox_read and resolution_dst_crs are expected in the same CRS"""
    # Compute affine transform out crs
    return rasterio.transform.from_origin(min(bbox_read[0], bbox_read[2]),
                                          max(bbox_read[1], bbox_read[3]),
                                          resolution_dst_crs[0], resolution_dst_crs[1])