import numpy as np
import pandas as pd
import xarray as xr

from japca.features.derive import (
    add_coastal_distance_proxy,
    add_moisture_flux,
    add_terrain_gradients,
    add_upslope_flow_proxy,
    add_vertical_shear,
    add_wind_speed,
)


def _base_dataset() -> xr.Dataset:
    time = pd.date_range("2023-01-01", periods=2, freq="6h")
    lat = [0.0, 1.0]
    lon = [0.0, 1.0]
    shape = (2, 2, 2)
    return xr.Dataset(
        {
            "u10": xr.DataArray(np.ones(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "v10": xr.DataArray(np.ones(shape) * 2, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "u700": xr.DataArray(np.ones(shape) * 4, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "v700": xr.DataArray(np.ones(shape) * 6, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "qv2": xr.DataArray(np.ones(shape) * 3, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "pwv": xr.DataArray(np.ones(shape) * 5, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
            "topo": xr.DataArray(np.arange(8).reshape(shape), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}),
        }
    )


def test_derived_feature_families():
    ds = _base_dataset()
    ds = add_wind_speed(ds)
    ds = add_moisture_flux(ds)
    ds = add_vertical_shear(ds)
    ds = add_terrain_gradients(ds)
    ds = add_upslope_flow_proxy(ds)
    ds = add_coastal_distance_proxy(ds)
    assert "wind_speed_10m" in ds
    assert "moisture_flux_u10" in ds
    assert "vertical_shear_speed" in ds
    assert "terrain_slope_lat" in ds
    assert "upslope_flow_10m" in ds
    assert "coastal_distance_proxy" in ds
