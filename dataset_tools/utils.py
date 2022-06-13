import sys
import os
from osgeo import ogr
from shapely.geometry import Point, Polygon, mapping
import requests
import subprocess
from osgeo import gdal, osr    
import logging

logger = logging.getLogger('sentinelloader')

def gmlToPolygon(gmlStr):
    footprint1 = ogr.CreateGeometryFromGML(gmlStr)
    coords = []
    if footprint1.GetGeometryCount() == 1:
        g0 = footprint1.GetGeometryRef(0)
        for i in range(0, g0.GetPointCount()):
            pt = g0.GetPoint(i)
            coords.append((pt[1], pt[0]))
    return Polygon(coords)


def downloadFile(url, filepath, user, password):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, "wb") as f:
        logger.debug("Downloading %s to %s" % (url, filepath))
        response = requests.get(url, auth=(user, password), stream=True)
        if response.status_code != 200:
            raise Exception("Could not download file. status=%s" % response.status_code)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)


def saveFile(filename, contents):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as fw:
        fw.write(contents)
        fw.flush()


def loadFile(filename):
    with open(filename, 'r') as fr:
        return fr.read()


def convertWGS84To3857(x, y):
    s1 = subprocess.check_output(["echo \"%s %s\" | cs2cs +init=epsg:4326 +to +init=epsg:3857" % (x,y)], shell=True)
    # s1 = !echo "{x} {y}" | cs2cs + init = epsg: 4326 + to + init = epsg: 3857
    s = s1.decode("utf-8").replace(" 0.00", "").split('\t')
    return (float(s[0]), float(s[1]))


def convertGeoJSONFromWGS84To3857(geojson):
    coords = []
    c = geojson['coordinates'][0]
    for co in list(c):
        coords.append(convertWGS84To3857(co[0], co[1]))
    geo = {
        'coordinates': ((tuple(coords)),),
        'type': geojson['type']
    }
    return geo

def saveGeoTiff(imageDataFloat, outputFile, geoTransform, projection):
    driver = gdal.GetDriverByName('GTiff')
    image_data = driver.Create(outputFile, imageDataFloat.shape[1], imageDataFloat.shape[0], 1, gdal.GDT_Float32)
    image_data.GetRasterBand(1).WriteArray(imageDataFloat)
    image_data.SetGeoTransform(geoTransform) 
    image_data.SetProjection(projection)
    image_data.FlushCache()
    image_data=None

from osgeo import gdal,ogr,osr

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append((y,x))
    return trans_coords

def getCoords(file):
    ds=gdal.Open(file)

    gt=ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext=GetExtent(gt,cols,rows)

    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    tgt_srs=osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)
    return geo_ext