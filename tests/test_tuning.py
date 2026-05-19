import numpy as np
import pandas as pd
import xarray as xr

from japca.baselines.hurdle import HurdleXGB
from japca.tuning.run import _predict_tabular_amount, _resolve_feature_columns, _split_hurdle_params


def _synthetic_eval_dataset() -> xr.Dataset:
    time = pd.date_range("2022-01-01", periods=2, freq="6h")
    lat = [0.0, 1.0]
    lon = [0.0, 1.0]
    shape = (2, 2, 2)
    return xr.Dataset(
        {
            "imerg_t": xr.DataArray(np.full(shape, 2.0), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "target_precip": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "target_occurrence": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
        }
    )


def test_split_hurdle_params_prefixed_and_threshold():
    classifier_params, regressor_params, threshold = _split_hurdle_params(
        {
            "classifier__max_depth": 6,
            "classifier__learning_rate": 0.05,
            "regressor__max_depth": 4,
            "occurrence_probability_threshold": 0.35,
        }
    )
    assert classifier_params == {"max_depth": 6, "learning_rate": 0.05}
    assert regressor_params == {"max_depth": 4}
    assert threshold == 0.35


def test_split_hurdle_params_backward_compatible():
    classifier_params, regressor_params, threshold = _split_hurdle_params(
        {
            "max_depth": 8,
            "n_estimators": 200,
        }
    )
    assert classifier_params == {"max_depth": 8, "n_estimators": 200}
    assert regressor_params == {"max_depth": 8, "n_estimators": 200}
    assert threshold is None


def test_predict_tabular_amount_persistence_hybrid_keeps_full_imerg_t_map():
    model = HurdleXGB(occurrence_probability_threshold=0.5)
    model.predict_occurrence_probability = lambda x: np.array([0.9, 0.2, 0.7, 0.4, 0.6, 0.1, 0.8, 0.3], dtype=float)  # type: ignore[method-assign]
    eval_ds = _synthetic_eval_dataset()
    x_eval = np.zeros((8, 1), dtype=float)
    pred_amount = _predict_tabular_amount("persistence_hybrid", model, x_eval, eval_ds)
    expected = np.full((2, 2, 2), 2.0)
    np.testing.assert_allclose(pred_amount, expected)


def test_predict_tabular_amount_persistence_hybrid_gated_masks_imerg_t():
    model = HurdleXGB(occurrence_probability_threshold=0.5)
    model.predict_occurrence_probability = lambda x: np.array([0.9, 0.2, 0.7, 0.4, 0.6, 0.1, 0.8, 0.3], dtype=float)  # type: ignore[method-assign]
    eval_ds = _synthetic_eval_dataset()
    x_eval = np.zeros((8, 1), dtype=float)
    pred_amount = _predict_tabular_amount("persistence_hybrid_gated", model, x_eval, eval_ds)
    expected = np.array(
        [
            [[2.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 0.0]],
        ]
    )
    np.testing.assert_allclose(pred_amount, expected)


def test_resolve_feature_columns_respects_feature_set_and_extra_aerosol():
    eval_ds = xr.Dataset(
        {
            "imerg_t": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_t_minus_6h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_t_minus_12h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_t_minus_18h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_diff_t_minus_6h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_diff_6h_minus_12h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_rolling_max_24h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "imerg_rolling_min_24h": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "pwv": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "qv2": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "moisture_flux_u10": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "moisture_flux_v10": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "moisture_flux_u700": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "moisture_flux_v700": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "topo": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "terrain_slope_lat": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "terrain_slope_lon": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "terrain_slope_magnitude": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "terrain_aspect": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "upslope_flow_10m": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "upslope_flow_700hpa": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "coastal_distance_proxy": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "totext": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "target_precip": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
            "target_occurrence": xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon")),
        }
    )
    feature_names = _resolve_feature_columns(
        eval_ds,
        {
            "feature_set": "minimal",
            "extra_feature_columns": ["totext"],
        },
    )
    assert "totext" in feature_names
    assert "pwv" in feature_names
    assert "topo" in feature_names
    assert feature_names.index("imerg_t") < feature_names.index("pwv")
    assert feature_names[-1] == "totext"
