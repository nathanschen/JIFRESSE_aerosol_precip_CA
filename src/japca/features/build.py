from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from japca.config import ensure_directory, load_paths_config
from japca.data.alignment import align_dataarrays_on_time, load_dataarray, valid_time_mask
from japca.data.manifest import DatasetManifest
from japca.features.derive import (
    add_coastal_distance_proxy,
    add_moisture_flux,
    add_terrain_gradients,
    add_upslope_flow_proxy,
    add_vertical_shear,
    add_wind_speed,
)
from japca.features.registry import resolve_feature_set


def _add_precip_history_features(
    dataset: xr.Dataset,
    base_precip: xr.DataArray,
    forecast_horizon_steps: int,
    cadence_hours: int,
) -> xr.Dataset:
    supervised_time = dataset.coords["time"]
    max_lag = 3
    lagged = {}
    for lag in range(max_lag + 1):
        name = "imerg_t" if lag == 0 else f"imerg_t_minus_{lag * cadence_hours}h"
        feature = base_precip.shift(time=lag).isel(time=slice(0, -forecast_horizon_steps))
        feature = feature.assign_coords(time=supervised_time)
        lagged[name] = feature
    dataset = dataset.assign(lagged)
    dataset["imerg_diff_t_minus_6h"] = dataset["imerg_t"] - dataset["imerg_t_minus_6h"]
    dataset["imerg_diff_6h_minus_12h"] = dataset["imerg_t_minus_6h"] - dataset["imerg_t_minus_12h"]
    dataset["imerg_rolling_max_24h"] = xr.concat(
        [
            dataset["imerg_t"],
            dataset["imerg_t_minus_6h"],
            dataset["imerg_t_minus_12h"],
            dataset["imerg_t_minus_18h"],
        ],
        dim="lag_component",
    ).max(dim="lag_component")
    dataset["imerg_rolling_min_24h"] = xr.concat(
        [
            dataset["imerg_t"],
            dataset["imerg_t_minus_6h"],
            dataset["imerg_t_minus_12h"],
            dataset["imerg_t_minus_18h"],
        ],
        dim="lag_component",
    ).min(dim="lag_component")
    return dataset


def build_feature_dataset(feature_set_name: str) -> xr.Dataset:
    paths = load_paths_config()
    manifest = DatasetManifest.from_config()
    feature_set = resolve_feature_set(feature_set_name)
    cadence_hours = int(paths["benchmark"]["cadence_hours"])
    forecast_horizon_steps = int(paths["benchmark"]["forecast_horizon_steps"])

    grid_spec = manifest.get(paths["canonical_grid"]["source_key"])
    grid = load_dataarray(grid_spec)
    target_lat = grid.coords["lat"]
    target_lon = grid.coords["lon"]

    requested = {"precip_target_raw", *feature_set.variables}
    arrays = {}
    for key in requested:
        spec = manifest.get(key)
        arrays[spec.build_name] = load_dataarray(spec, target_lat=target_lat, target_lon=target_lon)

    aligned = align_dataarrays_on_time(arrays)
    aligned_dataset = xr.Dataset(aligned)

    base_target = aligned_dataset["target_precip_raw"]
    predictor_slice = slice(0, -forecast_horizon_steps)
    target_slice = slice(forecast_horizon_steps, None)
    supervised = aligned_dataset.isel(time=predictor_slice).drop_vars("target_precip_raw")
    target = base_target.isel(time=target_slice).assign_coords(time=supervised.coords["time"])
    supervised["target_precip"] = target
    supervised["target_occurrence"] = (target >= 0.1).astype(np.int8)

    supervised = _add_precip_history_features(supervised, base_target, forecast_horizon_steps, cadence_hours)
    supervised = add_wind_speed(supervised)
    supervised = add_moisture_flux(supervised)
    supervised = add_vertical_shear(supervised)
    supervised = add_terrain_gradients(supervised)
    supervised = add_upslope_flow_proxy(supervised)
    supervised = add_coastal_distance_proxy(supervised)

    mask = valid_time_mask(supervised)
    supervised = supervised.sel(time=mask)
    supervised.attrs["feature_set"] = feature_set_name
    supervised.attrs["feature_group_membership"] = json.dumps(feature_set.group_membership)
    supervised.attrs["forecast_horizon_steps"] = forecast_horizon_steps
    supervised.attrs["cadence_hours"] = cadence_hours
    return supervised


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JAPCA benchmark feature dataset.")
    parser.add_argument("--feature-set", required=True, help="Feature set name from configs/feature_groups.yaml")
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    args = parser.parse_args()

    dataset = build_feature_dataset(args.feature_set)
    paths = load_paths_config()
    features_dir = ensure_directory(paths["paths"]["features_dir"])
    output_path = Path(args.output) if args.output else features_dir / f"{args.feature_set}.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    print(f"Saved feature dataset to {output_path}")


if __name__ == "__main__":
    main()
