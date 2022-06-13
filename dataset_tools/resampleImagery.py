# Needs to be run from root directory in order to use src submodules
from affine import Affine
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import rasterio.shutil as rasterio_shutil
import rasterio.rio.overview
import tempfile
import tqdm
import os

def add_overviews(rst_out, tile_size, verbose=False):
    """ Add overviews to be a cog and be displayed nicely in GIS software """

    overview_level = rasterio.rio.overview.get_maximum_overview_level(*rst_out.shape, tile_size)
    overviews = [2 ** j for j in range(1, overview_level + 1)]

    if verbose:
        print(f"Adding pyramid overviews to raster {overviews}")

    # Copied from https://github.com/cogeotiff/rio-cogeo/blob/master/rio_cogeo/cogeo.py#L274
    rst_out.build_overviews(overviews, Resampling.average)
    rst_out.update_tags(ns='rio_overview', resampling='nearest')
    tags = rst_out.tags()
    tags.update(OVR_RESAMPLING_ALG="NEAREST")
    rst_out.update_tags(**tags)
    rst_out._set_all_scales([rst_out.scales[b - 1] for b in rst_out.indexes])
    rst_out._set_all_offsets([rst_out.offsets[b - 1] for b in rst_out.indexes])


def save_cog(out_np, path_tiff_save, profile):
    """
    saves `out_np` np array as a COG GeoTIFF in path_tiff_save. profile is a dict with the geospatial info to be saved
    with the TiFF.

    :param out_np: 3D numpy array to save in CHW format
    :param path_tiff_save:
    :param profile: dict with profile to write geospatial info of the dataset: (crs, transform)
    """
    # Set blockysize, blockxsize
    for idx, b in enumerate(["blockysize","blockxsize"]):
        if b in profile:
            assert profile[b] <= 512, f"{b} is {profile[b]} must be <=512 to be displayed in GEE "
        else:
            profile[b] = min(512, out_np.shape[idx+1])

    dir_tiff_save = os.path.dirname(path_tiff_save)
    fileobj = tempfile.NamedTemporaryFile(dir=dir_tiff_save, suffix=".tif", delete=True)
    named_tempfile = fileobj.name
    fileobj.close()

    with rasterio.open(named_tempfile, "w", **profile) as rst_out:
        rst_out.write(out_np)
        add_overviews(rst_out, tile_size=profile["blockysize"])
        rasterio_shutil.copy(rst_out, path_tiff_save, copy_src_overviews=True, **profile)

    rasterio_shutil.delete(named_tempfile)
    return path_tiff_save

def resample(file, output_file, GSD, makeCOG=False):
    """
    inputs  - str file: file to be resampled
            - str output_file: path of output file
            - int GSD: required ground sampling distance
            - bool COG: ouput COG or not
    """
    with rasterio.open(file) as src:
        xres = yres = GSD
        # Determine the window to using bounds from the imagery.
        left, bottom, right, top = src.bounds
        dst_window = src.window(left, bottom, right, top)
        dst_width = int((right - left) / xres)
        dst_height = int((top - bottom) / yres)
        # Read into a channels x dst_width x dst_height (output dimensions)
        data = src.read(out_shape=(src.count, dst_height, dst_width),resampling=Resampling.cubic_spline)
        # Use the source's profile as a template for our output file.
        profile = src.profile
        profile['driver'] = 'GTiff'
        profile['width'] = dst_width
        profile['height'] = dst_height
        # Determine the affine transformation matrix
#         dst_transform = Affine(xres, 0.0, left,0.0, -yres, top)
        dst_transform = src.transform * src.transform.scale(src.width/dst_width,src.height/dst_height)
        profile['transform'] = dst_transform
        if makeCOG:
            save_cog(data, output_file, profile)
        else:
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(data)



def resample_dataset(root_dir,satellite,GSD,makeCOG=False):
    """
    inputs  - file: root directory ["/data/spacenet/train/"]
            - output_file: satellite ["planet","sentinel"]
            - GSD: required ground sampling distance in m
    """
    from src.datasources import spacenet7_index
    df = spacenet7_index()
    df_satellite = df.loc[satellite]
    scenes = (df_satellite
              .index
              .get_level_values('scene')  ## "scene" level of the multi-index
              .unique())  ## Unique names
    scenes = list(scenes)
    for scene in tqdm.tqdm(scenes):
        scene_df = df_satellite.query(f"scene=='{scene}'")
        output_dir = os.path.join(root_dir,scene,satellite+'_processed')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        paths = scene_df['path']
        filenames = scene_df['basename']
        for i in range(len(scene_df)):
            file = scene_df.iloc[i].path
            filename = scene_df.iloc[i].basename
            output_file = os.path.join(output_dir,filename)
            resample(file,output_file, GSD,makeCOG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resample Imagery')
    parser.add_argument('--satellite', required=True, help='Satellite Data to Resample')
    parser.add_argument('--root', default="/data/spacenet/train/", help='root_directory')
    parser.add_argument('--gsd', type=int, help='required ground sampling distance in meters')
    parser.add_argument('--COG', type=bool, help='output COG file')
    args = parser.parse_args()
    resample_dataset(args.root, args.satellite, args.gsd, args.COG)
