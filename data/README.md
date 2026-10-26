# Data Contract and Dataset Documentation

**Project:** Next Day Wildfire Spread Prediction via Tabular ML and Spatial Deep Learning  
**Team:** TensorTitans, Section C  
**Reference Paper:** Huot et al. (2022), *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data*, IEEE TGRS.

---

## 1. One-Page Data Contract

| Contract Dimension | Specification | Verification & Notes |
| :--- | :--- | :--- |
| **Prediction Unit** | $1\text{ km} \times 1\text{ km}$ spatial pixel on a given calendar observation day $t$. | Extracted from $64 \times 64$ patches centered around active wildfire events. |
| **Prediction Cutoff Time** | End of observation day $t$ (23:59:59 UTC). | Inputs strictly cannot access any observations timestamped $\ge t+1$. |
| **Target Label (`FireMask`)** | Binary indicator: $1$ if pixel burns on day $t+1$, $0$ otherwise. | Derived from next-day VIIRS active fire detection. Pixel value $1.0$ indicates active fire. Value $0.0$ indicates unburned; $-1.0$ (uncertain/masked) is filtered during training. |
| **Input Window** | Day $t$ current conditions and $t-1$ historical observations. | Weather, vegetation indices, drought, topography, and previous-day fire detection (`PrevFireMask` at day $t$). Target day $t+1$ is strictly excluded from feature extraction. |
| **Geographic Scope** | Contiguous United States (CONUS). | Wildfire events recorded between 2012 and 2020. |
| **Positive Class Definition** | Confirmed fire detection at day $t+1$ (`FireMask == 1.0`). | Severe class imbalance: positive burn pixels represent $< 0.3\%$ of eligible pixels. |
| **Splitting Protocol** | Spatially and temporally isolated partitions (Event-based separation). | Fire events and observation windows are partitioned so no fire event or adjacent spatial boundary spans train, validation, and test splits. |

---

## 2. Leakage Prevention Protocol

1. **Temporal Isolation**: No future weather, drought forecasts, or subsequent fire perimeter data are used. All dynamic features are computed strictly on or before day $t$.
2. **Spatial Isolation**: Train, validation, and test sets are partitioned by discrete wildfire events and geographic regions. No overlapping $64 \times 64$ patches or neighborhood filter windows span across split boundaries.
3. **Transformation Partitioning**: All imputation, scalers, encoders, and feature selection statistics are fitted strictly on the `train` split and subsequently transformed on `validation` and `test`.
4. **Frozen Test Set**: The test partition is locked until model architectures, hyperparameters, and decision thresholds are finalized using validation metrics.

---

## 3. Verified Dataset Schema & Feature Channels

The dataset provides 13 distinct 2D channels ($64 \times 64$ grid at 1 km resolution = 4,096 pixels per patch):

| Feature Channel | Family | Source / Sensor | Description & Units | Temporal Nature |
| :--- | :--- | :--- | :--- | :--- |
| `elevation` | Terrain | USGS SRTM / NED | Elevation above sea level (meters). | Static |
| `th` | Weather | GridMET / ERA5 | Wind direction at 10m (degrees clockwise from North). | Day $t$ Dynamic |
| `vs` | Weather | GridMET / ERA5 | Wind speed magnitude at 10m ($m/s$). | Day $t$ Dynamic |
| `tmmn` | Weather | GridMET / ERA5 | Minimum daily surface temperature (Kelvin). | Day $t$ Dynamic |
| `tmmx` | Weather | GridMET / ERA5 | Maximum daily surface temperature (Kelvin). | Day $t$ Dynamic |
| `sph` | Weather | GridMET / ERA5 | Specific humidity ($kg/kg$). | Day $t$ Dynamic |
| `pr` | Weather | GridMET / ERA5 | Accumulated daily precipitation ($mm$). | Day $t$ Dynamic |
| `pdsi` | Drought | GridMET | Palmer Drought Severity Index (soil moisture deficit). | Weekly Dynamic |
| `NDVI` | Vegetation | MODIS MOD13A2 | Normalized Difference Vegetation Index (greenness / biomass). | Bi-weekly Static |
| `population` | Anthropogenic | Gridded Population | Human population density ($people/km^2$). | Annual Static |
| `erc` | Drought / Fuel | GridMET | Energy Release Component (NFDRS fuel dryness / burn potential). | Daily Dynamic |
| `PrevFireMask` | Fire History | VIIRS Active Fire | Fire detection mask on day $t$ ($1 = \text{fire}, 0 = \text{no fire}$). | Day $t$ Lagged |
| **`FireMask`** | **Target** | **VIIRS Active Fire** | **Fire detection mask on day $t+1$ ($1 = \text{fire}, 0 = \text{no fire}$).** | **Day $t+1$ Target Label** |

---

## 4. Directory Structure & Storage Layout

```
data/
├── README.md               # This Data Contract & Documentation
├── raw/
│   └── archive/            # Raw TFRecord shards from Huot et al. (2022)
│       ├── next_day_wildfire_spread_train_00.tfrecord to _14.tfrecord  (15 shards, ~3.1 GB)
│       ├── next_day_wildfire_spread_eval_00.tfrecord and _01.tfrecord   (2 shards, ~400 MB)
│       └── next_day_wildfire_spread_test_00.tfrecord and _01.tfrecord   (2 shards, ~360 MB)
├── interim/                # Extracted and structured tabular samples (HDF5 / Parquet / CSV)
└── processed/              # Leakage-safe preprocessed features & split manifests
```
