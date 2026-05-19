import numpy as np
import pandas as pd
import torch
import xarray as xr

from japca.baselines.hurdle import HurdleXGB
from japca.models.architectures import TemporalUNet
from japca.models.training import GridForecastDataset


def _synthetic_dataset() -> xr.Dataset:
    time = pd.date_range("2021-01-01", periods=3, freq="6h")
    lat = [0.0, 1.0]
    lon = [0.0, 1.0]
    shape = (3, 2, 2)
    return xr.Dataset(
        {
            "pwv": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "imerg_t": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "imerg_t_minus_6h": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "imerg_t_minus_12h": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "imerg_t_minus_18h": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "target_precip": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "target_occurrence": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
        }
    )


def test_hurdle_xgb_smoke():
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y_train = np.array([0.0, 0.0, 1.0, 2.0], dtype=float)
    model = HurdleXGB()
    model.fit(x_train, y_train)
    pred_amount = model.predict_amount(np.array([[0.5], [2.5]], dtype=float))
    pred_prob = model.predict_occurrence_probability(np.array([[0.5], [2.5]], dtype=float))
    assert pred_amount.shape == (2,)
    assert pred_prob.shape == (2,)


def test_temporal_unet_smoke():
    dataset = _synthetic_dataset()
    source = GridForecastDataset(dataset, model_family="temporal_unet")
    inputs, occurrence, amount = source[0]
    model = TemporalUNet(in_channels=inputs.shape[0], hidden_channels=8)
    occ_logits, amount_pred = model(inputs.unsqueeze(0))
    assert occ_logits.shape == occurrence.unsqueeze(0).shape
    assert amount_pred.shape == amount.unsqueeze(0).shape
    assert isinstance(torch.sigmoid(occ_logits).detach().cpu().numpy(), np.ndarray)
