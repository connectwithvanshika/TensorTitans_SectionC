import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt


def calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0, 1.0
    idx = valid_indices[-1]
    return float(tpr[idx]), float(thresholds[idx])


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls + 1e-10),
        0.0
    )
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def evaluate_model_predictions(y_true, y_prob, threshold=None, target_fpr=0.05):
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    recall_at_fpr, thresh_at_fpr = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=target_fpr)
    recall_at_fpr_01, _ = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (len(y_true), 0, 0, 0)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_at_fpr_05": round(recall_at_fpr, 4),
        "recall_at_fpr_01": round(recall_at_fpr_01, 4),
        "optimal_threshold": round(threshold, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        }
    }


def plot_precision_recall_curves(models_dict, save_path="results/figures/precision_recall_curves.png"):
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in models_dict.items():
        pr_auc = average_precision_score(y_true, y_prob)
        p, r, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(r, p, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves on Wildfire Test Split", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
