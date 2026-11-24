"""Pure NumPy evaluation metrics and deterministic cross-validation splitters."""

import numpy as np


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC using exact Mann-Whitney U rank statistic."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_p)
    sorted_y = y_t[order]
    sorted_p = y_p[order]

    ranks = np.zeros(len(y_p), dtype=np.float64)
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    pos_rank_sum = float(np.sum(ranks[sorted_y == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.45,
) -> dict[str, float]:
    """Compute binary classification metrics prioritizing churn recall and ROC-AUC."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int64)

    tp = int(np.sum((y_t == 1) & (y_pred == 1)))
    fp = int(np.sum((y_t == 0) & (y_pred == 1)))
    tn = int(np.sum((y_t == 0) & (y_pred == 0)))
    fn = int(np.sum((y_t == 1) & (y_pred == 0)))

    total = max(tp + fp + tn + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    roc_auc = compute_roc_auc(y_t, y_p)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 regression metrics for 90-day CLV prediction."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    errors = y_t - y_p
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
    }


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic stratified K-Fold train/validation index pairs."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=np.int64)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        perm = rng.permutation(cls_idx)
        for i, idx in enumerate(perm):
            folds[i % n_splits].append(int(idx))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = set(range(len(y_arr)))
    for k in range(n_splits):
        val_idx = np.array(sorted(folds[k]), dtype=np.int64)
        train_idx = np.array(sorted(all_idx - set(folds[k])), dtype=np.int64)
        splits.append((train_idx, val_idx))
    return splits


def kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic K-Fold train/validation index pairs for regression."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    fold_arrays = np.array_split(perm, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_idx = np.sort(fold_arrays[k])
        train_idx = np.sort(np.concatenate([fold_arrays[j] for j in range(n_splits) if j != k]))
        splits.append((train_idx, val_idx))
    return splits
