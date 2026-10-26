import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from .feature_engineering import compute_spatial_neighborhood_features

FEATURE_NAMES = [
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc"
]
INPUT_CHANNELS = FEATURE_NAMES + ["PrevFireMask"]
TARGET_NAME = "FireMask"
ALL_CHANNELS = FEATURE_NAMES + ["PrevFireMask", TARGET_NAME]

GRID_SIZE = 64
NUM_PIXELS = GRID_SIZE * GRID_SIZE


def get_tfrecord_paths(split="train", raw_dir="data/raw/archive"):
    if split == "validation":
        split = "eval"

    if not os.path.exists(raw_dir) and os.path.exists(os.path.join("..", raw_dir)):
        raw_dir = os.path.join("..", raw_dir)

    pattern = os.path.join(raw_dir, f"next_day_wildfire_spread_{split}_*.tfrecord")
    files = sorted(glob.glob(pattern))
    return files


def get_feature_description():
    description = {}
    for name in ALL_CHANNELS:
        description[name] = tf.io.FixedLenFeature([NUM_PIXELS], tf.float32)
    return description


def parse_tfrecord_raw(example_proto):
    features = tf.io.parse_single_example(example_proto, get_feature_description())
    patch = {}
    for name in ALL_CHANNELS:
        patch[name] = tf.reshape(features[name], (GRID_SIZE, GRID_SIZE)).numpy()
    return patch


def iter_records(tfrecord_paths, max_records=None):
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    count = 0
    for raw_record in dataset:
        parsed = parse_tfrecord_raw(raw_record)
        yield parsed
        count += 1
        if max_records is not None and count >= max_records:
            break


def patch_to_tabular(patch, patch_id, filter_uncertain=True, sample_negatives_ratio=None, random_state=42):
    rows, cols = np.indices((GRID_SIZE, GRID_SIZE))
    center = GRID_SIZE / 2.0
    dist_to_center = np.sqrt((rows - center) ** 2 + (cols - center) ** 2) / (GRID_SIZE / 2.0)

    data = {
        "patch_id": patch_id,
        "row": rows.flatten(),
        "col": cols.flatten(),
        "dist_to_center": dist_to_center.flatten(),
    }
    for ch in ALL_CHANNELS:
        data[ch] = patch[ch].flatten()

    sum_3x3, sum_5x5 = compute_spatial_neighborhood_features(patch["PrevFireMask"])
    data["fire_neighbor_3x3_sum"] = sum_3x3.flatten()
    data["fire_neighbor_5x5_sum"] = sum_5x5.flatten()

    df = pd.DataFrame(data)

    if filter_uncertain:
        valid_mask = (df[TARGET_NAME] >= 0.0) & (df["PrevFireMask"] >= 0.0)
        df = df[valid_mask].copy()

    if sample_negatives_ratio is not None and len(df) > 0:
        pos_df = df[df[TARGET_NAME] == 1.0]
        neg_df = df[df[TARGET_NAME] == 0.0]

        n_pos = len(pos_df)
        if n_pos > 0:
            target_neg = int(max(n_pos * sample_negatives_ratio, 20))
            if len(neg_df) > target_neg:
                neg_df = neg_df.sample(n=target_neg, random_state=random_state)
            df = pd.concat([pos_df, neg_df], ignore_index=True)

    return df


def extract_split_tabular(split, max_patches=100, sample_negatives_ratio=10.0, random_state=42):
    paths = get_tfrecord_paths(split)
    if not paths:
        raise FileNotFoundError(f"No TFRecord files found for split='{split}'")

    dfs = []
    patch_idx = 0

    for patch in iter_records(paths, max_records=max_patches):
        df_patch = patch_to_tabular(
            patch=patch,
            patch_id=f"{split}_{patch_idx}",
            filter_uncertain=True,
            sample_negatives_ratio=sample_negatives_ratio,
            random_state=random_state + patch_idx
        )
        if len(df_patch) > 0:
            df_patch["split"] = split
            dfs.append(df_patch)
        patch_idx += 1

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def save_interim_dataset(df, name, interim_dir="data/interim"):
    os.makedirs(interim_dir, exist_ok=True)
    csv_path = os.path.join(interim_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
