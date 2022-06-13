# RobustMFSRforEO
Developing Robust MFSR for Earth Observation

## Environments

#### Pip

```
virtualenv --python=python3.8 ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```


#### Conda
Install [miniconda](https://pytorch.org/get-started/locally/#anaconda), like so:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
Create the environment:
```
conda env create -f environments/solarisenv.yml
pip install solaris
```

## Containers

#### Singularity
The image is in `/containers/mfsr.simg` .

For an interactive shell:
```
singularity shell -B /data:/data /containers/mfsr.simg
```

and to run a script:
```
singularity exec /containers/mfsr.simg python3 main.py --epochs 8
```

#### Docker
(tbc)



## Tests

```
export PYTHONPATH=.
pytest --cov-report term-missing --cov=src
```

## Dataset folder structure

```
spacenet
├── csvs
└── train
    └── L15-XXXE-XXXXN_XXXX_XXXX_XX
        ├── UDM_masks                # mask of clouds of planet imagery
        ├── images                   # Raw Planet Images
        ├── images_masked            # Planet images with cloud masked out
        ├── labels                   # GeoJSON labels of buildings and clouds (don’t worry about labels match)
        ├── masks                    # Labels in mask as geotiff form
        ├── sentinel                 # Sentinel imagery for that AOI
        ├── S2L2A                    # L2 processed Sentinel imagery (using this one)
        ├── sentinel_cloud           # GeoTiff of Sentinel Cloud Masks
        └── sentinel_processed       # resampled sentinel imagery to 10m
        └── sentinel_cloud_processed # resampled sentinel cloud imagery to 10m
        
```
