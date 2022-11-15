<img src="https://images.squarespace-cdn.com/content/v1/5b5740828ab72299c0747f05/1563553398375-M5CAV18Z00IGJ2GJMF9X/fdleuropeESA.png?format=1500w">

# Multi-spectral multi-image super-resolution of Sentinel-2 with radiometric consistency losses and its effect on building delineation

**Muhammed Razzak, Gonzalo Mateo-Garcia, Gurvan Lecuyer, Luis Gómez-Chova, Yarin Gal and Freddie Kalaitzis**

ISPRS Journal of Photogrammetry and Remote Sensing, vol. 195, pp. 1–13, Jan. 2023, DOI: [10.1016/j.isprsjprs.2022.10.019](https://doi.org/10.1016/j.isprsjprs.2022.10.019).

This repository contains the source code for [Multi-spectral multi-image super-resolution of Sentinel-2 with radiometric consistency losses and its effect on building delineation](https://doi.org/10.1016/j.isprsjprs.2022.10.019) and code for estimating uncertainty of models. Dataset is hosted on [SpaceML.org](https://spaceml.org/repo/project/61c0a1b9ff8868000dfb79e1).

## Set up environment

```
virtualenv --python=python3.8 ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Dataset 

Data is stored in `gs://fdl_misr/` public bucket. To download the data you can use `gsutil`. 

This [notebook](./notebooks/explore_dataset_fdl_misr.ipynb) may help you explore the dataset.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ESA-PhiLab/RobustMFSRforEO/blob/main/notebooks/explore_dataset_fdl_misr.ipynb)

The dataset follows this folder structure:
```
gs://fdl_misr/
└── public
    └── L15-XXXE-XXXXN_XXXX_XXXX_XX
        ├── UDM_masks                # mask of clouds of planet imagery
        ├── images                   # Raw Planet Images
        ├── labels                   # GeoJSON labels of buildings and clouds of Planet images
        ├── masks                    # SLC labels for Sentinel-2 L2A in geotiff
        └── S2L2A                    # L2 processed Sentinel imagery
        
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

## Acknowledgments

This work has been enabled by [Frontier Development Lab (FDL) Europe](https://fdleurope.org/), a public partnership between the European Space Agency (ESA) at Phi-Lab (ESRIN), Trillium Technologies and the University of Oxford; the project has been also supported by Google Cloud. G.M.G. and L.G.C. are funded by the Spanish Ministry of Science and Innovation, Spain (project PID2019-109026RB-I00). The authors would like to thank the support of James Parr and Jodie Hughes from the Trillium team and to Nicolas Longépé from ESA PhiLab for discussions and comments throughout the development of this work.
