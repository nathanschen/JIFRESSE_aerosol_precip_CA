import numpy as np
import pandas as pd
import xarray as xr

from japca.data.alignment import align_dataarrays_on_time, build_milestone_one_masks, canonicalize_coords, validate_grid_shape
from japca.features.build import _select_evenly_spaced_positions, _select_supervised_indices


def test_canonicalize_and_validate_grid_shape():
    da = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2023-01-01", periods=2, freq="6h"),
            "lat": [2.0, 1.0],
            "lon": [5.0, 4.0],
        },
    )
    normalized = canonicalize_coords(da)
    assert normalized.dims == ("time", "lat", "lon")
    assert list(normalized.lat.values) == [1.0, 2.0]
    assert list(normalized.lon.values) == [4.0, 5.0]
    assert validate_grid_shape(normalized, 2, 2)


def test_align_dataarrays_and_masks():
    common_time = pd.date_range("2022-12-31 18:00", periods=4, freq="6h")
    arr1 = xr.DataArray(np.ones((4, 1, 1)), dims=("time", "lat", "lon"), coords={"time": common_time, "lat": [1.0], "lon": [1.0]}).rename("a")
    arr2 = xr.DataArray(np.ones((3, 1, 1)), dims=("time", "lat", "lon"), coords={"time": common_time[1:], "lat": [1.0], "lon": [1.0]}).rename("b")
    aligned = align_dataarrays_on_time({"a": arr1, "b": arr2})
    assert aligned["a"].sizes["time"] == 3
    masks = build_milestone_one_masks(pd.date_range("2017-01-01", periods=7, freq="365D"))
    assert masks.train_mask.sum() >= 1


def test_split_balanced_sampling_helpers():
    assert _select_evenly_spaced_positions(5, 2).tolist() == [0, 4]

    common_time = pd.date_range("2017-01-01", "2023-12-31 18:00", freq="6h")
    sampled = _select_supervised_indices(
        common_time=common_time,
        forecast_horizon_steps=1,
        max_lag=3,
        sampling_cfg={"train_per_year": 2, "dev": 3, "test": 4},
    )
    sampled_years = common_time[sampled].year
    counts = {year: int((sampled_years == year).sum()) for year in sorted(set(sampled_years))}
    assert counts[2017] == 2
    assert counts[2021] == 2
    assert counts[2022] == 3
    assert counts[2023] == 4
