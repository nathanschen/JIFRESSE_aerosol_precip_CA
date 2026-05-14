import numpy as np

from japca.metrics.weather import brier_score, csi, far, fractions_skill_score, pod, rainy_pixel_bias, rainy_pixel_mae


def test_binary_metrics_and_amount_metrics():
    truth = np.array([[1, 0], [1, 1]])
    pred = np.array([[1, 0], [0, 1]])
    assert csi(truth, pred) == 2 / 3
    assert pod(truth, pred) == 2 / 3
    assert far(truth, pred) == 0.0
    assert np.isclose(brier_score(truth, pred.astype(float)), 0.25)

    amount_truth = np.array([[0.0, 2.0], [4.0, 0.0]])
    amount_pred = np.array([[0.0, 1.0], [5.0, 0.0]])
    assert np.isclose(rainy_pixel_mae(amount_truth, amount_pred, rainy_threshold=0.1), 1.0)
    assert np.isclose(rainy_pixel_bias(amount_truth, amount_pred, rainy_threshold=0.1), 0.0)
    assert 0.0 <= fractions_skill_score(truth, pred, window=1) <= 1.0
