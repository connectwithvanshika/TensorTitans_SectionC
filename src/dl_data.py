import os
import glob
import numpy as np
import tensorflow as tf
import keras

RAW_NAMES = ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask"]
CHANNELS = ["elevation", "vs", "wind_x", "wind_y", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask"]


def convert_split(split, raw_dir, out_dir):
    names = RAW_NAMES + ["FireMask"]
    description = {}
    for name in names:
        description[name] = tf.io.FixedLenFeature([64 * 64], tf.float32)

    def parse(example):
        parsed = tf.io.parse_single_example(example, description)
        x = tf.stack([tf.reshape(parsed[name], (64, 64)) for name in RAW_NAMES], axis=-1)
        y = tf.reshape(parsed["FireMask"], (64, 64))
        return x, y

    files = sorted(glob.glob(os.path.join(raw_dir, f"next_day_wildfire_spread_{split}_*.tfrecord")))
    xs = []
    ys = []
    for x, y in tf.data.TFRecordDataset(files).map(parse).batch(500):
        xs.append(x.numpy())
        ys.append(y.numpy().astype(np.int8))
    np.save(os.path.join(out_dir, f"{split}_X.npy"), np.concatenate(xs))
    np.save(os.path.join(out_dir, f"{split}_Y.npy"), np.concatenate(ys))


def make_channels(X):
    th = np.deg2rad(X[..., 1])
    vs = X[..., 2]
    wind_x = -vs * np.sin(th)
    wind_y = -vs * np.cos(th)
    prev = np.clip(X[..., 11], 0, 1)
    out = np.stack([X[..., 0], vs, wind_x, wind_y, X[..., 3], X[..., 4], X[..., 5], X[..., 6],
                    X[..., 7], X[..., 8], X[..., 9], X[..., 10], prev], axis=-1)
    return out.astype(np.float32)


def load_split(data_dir, split, limit=None):
    raw = np.load(os.path.join(data_dir, f"{split}_X.npy"), mmap_mode="r")[:limit]
    Y = np.load(os.path.join(data_dir, f"{split}_Y.npy"))[:limit]
    parts = []
    for i in range(0, len(raw), 1000):
        parts.append(make_channels(raw[i:i + 1000]).astype(np.float16))
    return np.concatenate(parts), Y


def fit_normalizer(train_X):
    low = []
    high = []
    mean = []
    std = []
    for c, name in enumerate(CHANNELS):
        sample = train_X[::10, ::4, ::4, c].astype(np.float32).ravel()
        if name == "PrevFireMask":
            low.append(0.0)
            high.append(1.0)
            mean.append(0.0)
            std.append(1.0)
        else:
            lo, hi = np.percentile(sample, [0.1, 99.9])
            clipped = np.clip(sample, lo, hi)
            low.append(lo)
            high.append(hi)
            mean.append(clipped.mean())
            std.append(clipped.std() + 1e-6)
    return {"low": np.array(low, dtype=np.float32), "high": np.array(high, dtype=np.float32),
            "mean": np.array(mean, dtype=np.float32), "std": np.array(std, dtype=np.float32)}


def normalize(X, norm):
    out = np.empty(X.shape, dtype=np.float16)
    for i in range(0, len(X), 1000):
        part = X[i:i + 1000].astype(np.float32)
        out[i:i + 1000] = (np.clip(part, norm["low"], norm["high"]) - norm["mean"]) / norm["std"]
    return out


class FireData(keras.utils.PyDataset):
    def __init__(self, X, Y, batch_size=32, augment=False, pos_weight=1.0, drop_channels=None, fix_wind=True):
        super().__init__()
        self.fix_wind = fix_wind
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.augment = augment
        self.pos_weight = pos_weight
        self.keep = [i for i, name in enumerate(CHANNELS) if name not in (drop_channels or [])]
        self.order = np.arange(len(X))
        if augment:
            np.random.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = np.sort(self.order[i * self.batch_size:(i + 1) * self.batch_size])
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1, :].copy()
                y = y[:, :, ::-1].copy()
                if self.fix_wind:
                    x[..., 2] = -x[..., 2]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :, :].copy()
                y = y[:, ::-1, :].copy()
                if self.fix_wind:
                    x[..., 3] = -x[..., 3]
        w = np.where(y == 1, self.pos_weight, 1.0)
        w = np.where(y < 0, 0.0, w).astype(np.float32)
        y = np.clip(y, 0, 1)
        return x[..., self.keep], y[..., None], w

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.order)
