import pandas as pd
from shapely.geometry import Polygon
from .utils import getCoords
import os
import shutil

df = pd.read_pickle('spacenet202001.pkl')
root = "/scratch-ssd/oatml/data/spacenet/train/"
from .Sentinel2Loader import Sentinel2Loader
sl = Sentinel2Loader('/scratch-ssd/oatml/muhzak/sentinelcache', 
                    'mtrazzak', 'zivhu4-sortud-kuJxiv',
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=False, cloudCoverage=(0,100))
                    
for i in range(len(df)):
    area = Polygon(getCoords(df.iloc[i].image))
    geoTiffs = sl.getRegionHistory(area, 'TCI','10m', '2020-01-01', '2020-01-31', daysStep=1, aoi=df.iloc[i].AOI)
    if not os.path.exists(root+str(df.iloc[i].AOI)+'/sentinel'):
        os.makedirs(root+str(df.iloc[i].AOI)+'/sentinel')
    for geoTiff in geoTiffs:
        name = os.path.basename(geoTiff)
        shutil.move(geoTiff, root+str(df.iloc[i].AOI)+'/sentinel/'+name)
    print(str(i+1)+"/"+str(len(df)))