import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from sklearn.preprocessing import StandardScaler

GRID_SIZE = 64

ABLATION_GROUPS = {
    "Group_A": [
        "tmmn", "tmmx", "sph", "vs", "th", "pr"
    ],
    "Group_B": [
        "tmmn", "tmmx", "sph", "vs", "th", "pr",
        "PrevFireMask"
    ],
    "Group_C": [
        "tmmn", "tmmx", "sph", "vs", "th", "pr",
        "PrevFireMask",
        "elevation", "NDVI", "pdsi", "erc", "population"
    ],
    "Group_D": [
        "tmmn", "tmmx", "sph", "vs", "th", "pr",
        "PrevFireMask",
        "elevation", "NDVI", "pdsi", "erc", "population",
        "temp_range", "wind_u", "wind_v", "drought_fuel_index",
        "fire_neighbor_3x3_sum", "fire_neighbor_5x5_sum"
    ]
}


def compute_spatial_neighborhood_features(prev_fire_mask_2d):
    kernel_3x3 = np.ones((3, 3), dtype=np.float32)
    kernel_3x3[1, 1] = 0.0

    kernel_5x5 = np.ones((5, 5), dtype=np.float32)
    kernel_5x5[2, 2] = 0.0

    binary_fire = np.clip(prev_fire_mask_2d, 0.0, 1.0)

    sum_3x3 = convolve(binary_fire, kernel_3x3, mode="constant", cval=0.0)
    sum_5x5 = convolve(binary_fire, kernel_5x5, mode="constant", cval=0.0)
    return sum_3x3, sum_5x5


def engineer_tabular_features(df):
    df = df.copy()

    if "tmmx" in df.columns and "tmmn" in df.columns:
        df["temp_range"] = df["tmmx"] - df["tmmn"]

    if "vs" in df.columns and "th" in df.columns:
        rad = np.radians(df["th"])
        df["wind_u"] = -df["vs"] * np.sin(rad)
        df["wind_v"] = -df["vs"] * np.cos(rad)

    if "erc" in df.columns and "pdsi" in df.columns:
        df["drought_fuel_index"] = df["erc"] * (np.maximum(-df["pdsi"], 0.0) + 1.0)

    if "fire_neighbor_3x3_sum" not in df.columns:
        if "patch_id" in df.columns and "row" in df.columns and "col" in df.columns:
            neighbor_3x3_list = []
            neighbor_5x5_list = []
            for patch_id, patch_group in df.groupby("patch_id"):
                grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                for _, r in patch_group.iterrows():
                    grid[int(r["row"]), int(r["col"])] = max(r.get("PrevFireMask", 0.0), 0.0)
                n3, n5 = compute_spatial_neighborhood_features(grid)
                for _, r in patch_group.iterrows():
                    neighbor_3x3_list.append(n3[int(r["row"]), int(r["col"])])
                    neighbor_5x5_list.append(n5[int(r["row"]), int(r["col"])])
            df["fire_neighbor_3x3_sum"] = neighbor_3x3_list
            df["fire_neighbor_5x5_sum"] = neighbor_5x5_list
        else:
            df["fire_neighbor_3x3_sum"] = df.get("PrevFireMask", 0.0)
            df["fire_neighbor_5x5_sum"] = df.get("PrevFireMask", 0.0) * 2.0

    return df


def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(0.0))
    return scaler


def transform_features(df, scaler, feature_cols):
    return scaler.transform(df[feature_cols].fillna(0.0))
