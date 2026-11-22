import numpy as np
import keras
from keras import layers

from .dl_data import CHANNELS, FireData


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(n_channels=13, base=16, dropout=0.2):
    inputs = keras.Input((64, 64, n_channels))
    c1 = conv_block(inputs, base)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base * 4)
    p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, base * 8)
    b = layers.Dropout(dropout)(b)
    u3 = layers.UpSampling2D()(b)
    c4 = conv_block(layers.Concatenate()([u3, c3]), base * 4)
    u2 = layers.UpSampling2D()(c4)
    c5 = conv_block(layers.Concatenate()([u2, c2]), base * 2)
    u1 = layers.UpSampling2D()(c5)
    c6 = conv_block(layers.Concatenate()([u1, c1]), base)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return keras.Model(inputs, outputs, name="unet")


def train_unet(train_X, train_Y, val_X, val_Y, epochs=25, pos_weight=3.0, augment=True,
               loss="bce", drop_channels=None, fix_wind=True, patience=5, seed=42, verbose=2):
    keras.utils.set_random_seed(seed)
    n_channels = len(CHANNELS) - len(drop_channels or [])
    model = build_unet(n_channels)
    if loss == "focal":
        loss_fn = keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    else:
        loss_fn = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4), loss=loss_fn,
                  weighted_metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    train_data = FireData(train_X, train_Y, 32, augment, pos_weight, drop_channels, fix_wind)
    val_data = FireData(val_X, val_Y, 64, False, 1.0, drop_channels)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=2),
    ]
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks, verbose=verbose)
    return model, history.history


def predict_maps(model, X, drop_channels=None):
    keep = [i for i, name in enumerate(CHANNELS) if name not in (drop_channels or [])]
    probs = []
    for i in range(0, len(X), 256):
        x = X[i:i + 256].astype(np.float32)[..., keep]
        probs.append(model.predict(x, batch_size=64, verbose=0)[..., 0])
    return np.concatenate(probs)
