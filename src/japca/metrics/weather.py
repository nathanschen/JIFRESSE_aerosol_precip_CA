from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score


def _flatten(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    truth = _flatten(y_true).astype(bool)
    pred = _flatten(y_pred).astype(bool)
    hits = int(np.logical_and(truth, pred).sum())
    misses = int(np.logical_and(truth, ~pred).sum())
    false_alarms = int(np.logical_and(~truth, pred).sum())
    return hits, misses, false_alarms


def csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits, misses, false_alarms = confusion_counts(y_true, y_pred)
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom else 0.0


def pod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits, misses, _ = confusion_counts(y_true, y_pred)
    denom = hits + misses
    return float(hits / denom) if denom else 0.0


def far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits, _, false_alarms = confusion_counts(y_true, y_pred)
    denom = hits + false_alarms
    return float(false_alarms / denom) if denom else 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    truth = _flatten(y_true).astype(float)
    prob = _flatten(y_prob).astype(float)
    return float(np.mean((prob - truth) ** 2))


def rainy_pixel_mae(y_true_amount: np.ndarray, y_pred_amount: np.ndarray, rainy_threshold: float = 0.1) -> float:
    true_amount = np.asarray(y_true_amount)
    pred_amount = np.asarray(y_pred_amount)
    mask = true_amount >= rainy_threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(true_amount[mask] - pred_amount[mask])))


def rainy_pixel_bias(y_true_amount: np.ndarray, y_pred_amount: np.ndarray, rainy_threshold: float = 0.1) -> float:
    true_amount = np.asarray(y_true_amount)
    pred_amount = np.asarray(y_pred_amount)
    mask = true_amount >= rainy_threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(pred_amount[mask] - true_amount[mask]))


def _window_fraction(field: np.ndarray, window: int) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    padded = np.pad(field, ((window // 2, window // 2), (window // 2, window // 2)), mode="constant")
    out = np.zeros_like(field, dtype=float)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            patch = padded[i : i + window, j : j + window]
            out[i, j] = patch.mean()
    return out


def fractions_skill_score(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, window: int) -> float:
    truth = np.asarray(y_true_binary)
    pred = np.asarray(y_pred_binary)
    if truth.ndim == 2:
        truth = truth[np.newaxis, ...]
        pred = pred[np.newaxis, ...]
    numerator = 0.0
    denominator = 0.0
    for truth_frame, pred_frame in zip(truth, pred):
        truth_frac = _window_fraction(truth_frame.astype(float), window)
        pred_frac = _window_fraction(pred_frame.astype(float), window)
        numerator += np.sum((truth_frac - pred_frac) ** 2)
        denominator += np.sum(truth_frac ** 2 + pred_frac ** 2)
    if denominator == 0:
        return 1.0
    return float(1.0 - numerator / denominator)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: Iterable[float]) -> list[dict[str, float]]:
    truth = _flatten(y_true).astype(float)
    prob = _flatten(y_prob).astype(float)
    edges = np.asarray(list(bins), dtype=float)
    result = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not np.any(mask):
            result.append({"bin_left": float(left), "bin_right": float(right), "count": 0, "mean_prob": 0.0, "event_rate": 0.0})
            continue
        result.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(mask.sum()),
                "mean_prob": float(prob[mask].mean()),
                "event_rate": float(truth[mask].mean()),
            }
        )
    return result


def composite_score(metrics: dict, ranking_weights: dict[str, float]) -> float:
    spatial = float(np.mean(list(metrics["fss"].values()))) if metrics.get("fss") else 0.0
    threshold = float(np.mean([item["csi"] for item in metrics["threshold_metrics"].values()])) if metrics.get("threshold_metrics") else 0.0
    occurrence = float(metrics.get("pr_auc", 0.0)) - float(metrics.get("brier_score", 0.0))
    amount = -float(metrics.get("rainy_pixel_mae", 0.0)) - abs(float(metrics.get("rainy_pixel_bias", 0.0)))
    return (
        ranking_weights["spatial_fss"] * spatial
        + ranking_weights["threshold_csi"] * threshold
        + ranking_weights["occurrence_quality"] * occurrence
        + ranking_weights["amount_quality"] * amount
    )


def evaluate_forecast(
    y_true_amount: np.ndarray,
    y_pred_amount: np.ndarray,
    y_occurrence_prob: np.ndarray,
    thresholds: Iterable[float],
    fss_windows: Iterable[int],
    ranking_weights: dict[str, float],
    reliability_bin_edges: Iterable[float],
) -> dict:
    true_amount = np.asarray(y_true_amount)
    pred_amount = np.asarray(y_pred_amount)
    prob = np.asarray(y_occurrence_prob)
    occ_truth = true_amount >= 0.1
    pr_auc = float(average_precision_score(_flatten(occ_truth), _flatten(prob)))
    metrics = {
        "pr_auc": pr_auc,
        "brier_score": brier_score(occ_truth, prob),
        "mae": float(np.mean(np.abs(pred_amount - true_amount))),
        "rmse": float(np.sqrt(np.mean((pred_amount - true_amount) ** 2))),
        "rainy_pixel_mae": rainy_pixel_mae(true_amount, pred_amount),
        "rainy_pixel_bias": rainy_pixel_bias(true_amount, pred_amount),
        "threshold_metrics": {},
        "fss": {},
        "reliability": reliability_bins(occ_truth, prob, reliability_bin_edges),
    }
    for threshold in thresholds:
        truth_bin = true_amount >= threshold
        pred_bin = pred_amount >= threshold
        metrics["threshold_metrics"][f"{threshold:g}"] = {
            "csi": csi(truth_bin, pred_bin),
            "pod": pod(truth_bin, pred_bin),
            "far": far(truth_bin, pred_bin),
        }
    for window in fss_windows:
        metrics["fss"][str(window)] = fractions_skill_score(occ_truth, pred_amount >= 0.1, window)
    metrics["composite_score"] = composite_score(metrics, ranking_weights)
    return metrics
