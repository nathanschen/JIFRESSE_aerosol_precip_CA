from __future__ import annotations

import numpy as np
import xarray as xr


def add_wind_speed(dataset: xr.Dataset) -> xr.Dataset:
    if {"u10", "v10"}.issubset(dataset.data_vars):
        dataset["wind_speed_10m"] = np.sqrt(dataset["u10"] ** 2 + dataset["v10"] ** 2)
    if {"u700", "v700"}.issubset(dataset.data_vars):
        dataset["wind_speed_700hpa"] = np.sqrt(dataset["u700"] ** 2 + dataset["v700"] ** 2)
    return dataset


def add_moisture_flux(dataset: xr.Dataset) -> xr.Dataset:
    if {"qv2", "u10", "v10"}.issubset(dataset.data_vars):
        dataset["moisture_flux_u10"] = dataset["qv2"] * dataset["u10"]
        dataset["moisture_flux_v10"] = dataset["qv2"] * dataset["v10"]
    if {"pwv", "u700", "v700"}.issubset(dataset.data_vars):
        dataset["moisture_flux_u700"] = dataset["pwv"] * dataset["u700"]
        dataset["moisture_flux_v700"] = dataset["pwv"] * dataset["v700"]
    return dataset


def add_vertical_shear(dataset: xr.Dataset) -> xr.Dataset:
    if {"u10", "v10", "u700", "v700"}.issubset(dataset.data_vars):
        du = dataset["u700"] - dataset["u10"]
        dv = dataset["v700"] - dataset["v10"]
        dataset["vertical_shear_u"] = du
        dataset["vertical_shear_v"] = dv
        dataset["vertical_shear_speed"] = np.sqrt(du ** 2 + dv ** 2)
    return dataset


def add_terrain_gradients(dataset: xr.Dataset) -> xr.Dataset:
    if "topo" not in dataset:
        return dataset
    topo = dataset["topo"]
    topo_slice = topo.isel(time=0) if "time" in topo.dims else topo
    lat_grad, lon_grad = np.gradient(topo_slice.values)
    grad_lat = xr.DataArray(lon_grad * 0 + lat_grad, coords=topo_slice.coords, dims=topo_slice.dims)
    grad_lon = xr.DataArray(lat_grad * 0 + lon_grad, coords=topo_slice.coords, dims=topo_slice.dims)
    if "time" in topo.dims:
        grad_lat = grad_lat.expand_dims(time=topo.coords["time"]).transpose("time", "lat", "lon")
        grad_lon = grad_lon.expand_dims(time=topo.coords["time"]).transpose("time", "lat", "lon")
    dataset["terrain_slope_lat"] = grad_lat
    dataset["terrain_slope_lon"] = grad_lon
    dataset["terrain_slope_magnitude"] = np.sqrt(grad_lat ** 2 + grad_lon ** 2)
    dataset["terrain_aspect"] = np.arctan2(grad_lat, grad_lon)
    return dataset


def add_upslope_flow_proxy(dataset: xr.Dataset) -> xr.Dataset:
    if {"u10", "v10", "terrain_slope_lon", "terrain_slope_lat"}.issubset(dataset.data_vars):
        dataset["upslope_flow_10m"] = dataset["u10"] * dataset["terrain_slope_lon"] + dataset["v10"] * dataset["terrain_slope_lat"]
    if {"u700", "v700", "terrain_slope_lon", "terrain_slope_lat"}.issubset(dataset.data_vars):
        dataset["upslope_flow_700hpa"] = dataset["u700"] * dataset["terrain_slope_lon"] + dataset["v700"] * dataset["terrain_slope_lat"]
    return dataset


def add_coastal_distance_proxy(dataset: xr.Dataset) -> xr.Dataset:
    if "lon" not in dataset.coords:
        return dataset
    lon = dataset.coords["lon"]
    west = float(lon.min())
    east = float(lon.max())
    denom = max(east - west, 1e-6)
    coast = (lon - west) / denom
    proxy = xr.DataArray(
        np.broadcast_to(coast.values, (dataset.sizes["lat"], dataset.sizes["lon"])),
        dims=("lat", "lon"),
        coords={"lat": dataset.coords["lat"], "lon": lon},
    )
    if "time" in dataset.dims:
        proxy = proxy.expand_dims(time=dataset.coords["time"]).transpose("time", "lat", "lon")
    dataset["coastal_distance_proxy"] = proxy
    return dataset
