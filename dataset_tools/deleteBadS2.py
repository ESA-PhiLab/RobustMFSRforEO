# Needs to be placed in root directory to access src submodules
import os
import pandas as pd
import rasterio
from src.datasources import spacenet7_index
df = spacenet7_index()
df_sentinel = df.query(f"sat=='sentinel'")['path']
sentinel_list = list(df_sentinel)
invalid_files = []
# Find invalid files
for file in sentinel_list:
    try:
        src = rasterio.open(file).read()
        maxi, mini = src.max(), src.min()
        if (mini == 255) and (maxi ==255):
            invalid_files += [file]
    except:
        invalid_files += [file]

# Delete the files
for invalid in invalid_files:
    if os.path.exists(invalid):
        os.remove(invalid)
    else:
        print("The file does not exist")
        
print("%d of the %d Sentinel 2 images are invalid and have been deleted." % (len(invalid_files), len(sentinel_list)))