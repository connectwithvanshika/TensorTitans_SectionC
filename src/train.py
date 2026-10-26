import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb


class PersistenceBaseline:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, prev_fire_mask):
        p = np.clip(prev_fire_mask, 0.0, 1.0).astype(float)
        return np.vstack([1.0 - p, p]).T

    def predict(self, prev_fire_mask):
        return (prev_fire_mask > 0.5).astype(int)


def train_logistic_regression(X_train, y_train, class_weight="balanced", random_state=42):
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return clf


def train_lightgbm_model(X_train, y_train, X_val=None, y_val=None, params=None, random_state=42):
    default_params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10.0,
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)

    clf = lgb.LGBMClassifier(**default_params)

    if X_val is not None and y_val is not None:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )
    else:
        clf.fit(X_train, y_train)

    return clf


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None, random_state=42):
    default_params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10.0,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1
    }
    if params:
        default_params.update(params)

    clf = xgb.XGBClassifier(**default_params)

    if X_val is not None and y_val is not None:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        clf.fit(X_train, y_train)

    return clf
