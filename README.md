# Multi-spectral multi-image super-resolution of Sentinel-2 with radiometric consistency losses and its effect on building delineation

**Muhammed Razzak, Gonzalo Mateo-Garcia, Gurvan Lecuyer, Luis Gómez-Chova, Yarin Gal and Freddie Kalaitzis**

This repository contains the source code for [Multi-spectral multi-image super-resolution of Sentinel-2 with radiometric consistency losses and its effect on building delineation](https://doi.org/10.1016/j.isprsjprs.2022.10.019) and code for estimating uncertainty of models. Dataset is hosted on [SpaceML.org](https://spaceml.org/repo/project/61c0a1b9ff8868000dfb79e1).

## Environments

#### Pip

```
virtualenv --python=python3.8 ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

```
## Dataset folder structure

```
gs://fdl_misr/
└── public
    └── L15-XXXE-XXXXN_XXXX_XXXX_XX
        ├── UDM_masks                # mask of clouds of planet imagery
        ├── images                   # Raw Planet Images
        ├── labels                   # GeoJSON labels of buildings and clouds (don’t worry about labels match)
        ├── masks                    # Labels in mask as geotiff form
        └── S2L2A                    # L2 processed Sentinel imagery (using this one)        
        
```

## Cite

If you find this work useful please cite:

```
@article{razzak_multi-spectral_2023,
	title = {Multi-spectral multi-image super-resolution of {Sentinel}-2 with radiometric consistency losses and its effect on building delineation},
	volume = {195},
	issn = {0924-2716},
	url = {https://www.sciencedirect.com/science/article/pii/S0924271622002878},
	doi = {10.1016/j.isprsjprs.2022.10.019},
	language = {en},
	urldate = {2022-11-14},
	journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
	author = {Razzak, Muhammed T. and Mateo-García, Gonzalo and Lecuyer, Gurvan and Gómez-Chova, Luis and Gal, Yarin and Kalaitzis, Freddie},
	month = jan,
	year = {2023},
	pages = {1--13},
}
```
