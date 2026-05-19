from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from japca.config import ensure_directory, load_paths_config
from japca.data.alignment import align_dataarrays_on_time, build_milestone_one_masks, canonicalize_coords, load_dataarray, valid_time_mask
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


def _select_evenly_spaced_positions(size: int, count: int) -> np.ndarray:
    if size <= 0 or count <= 0:
        return np.array([], dtype=int)
    if count >= size:
        return np.arange(size, dtype=int)
    positions = np.linspace(0, size - 1, num=count)
    return np.unique(np.round(positions).astype(int))


def _load_time_index(spec) -> pd.DatetimeIndex | None:
    dataset = xr.open_dataset(spec.path)
    if spec.variable not in dataset:
        raise KeyError(f"{spec.variable} not found in {spec.path}")
    da = canonicalize_coords(dataset[spec.variable])
    if "time" not in da.coords:
        return None
    return pd.DatetimeIndex(da["time"].values)


def _select_supervised_indices(
    common_time: pd.DatetimeIndex,
    forecast_horizon_steps: int,
    max_lag: int,
    sampling_cfg: dict,
) -> np.ndarray:
    candidate_indices = np.arange(max_lag, len(common_time) - forecast_horizon_steps, dtype=int)
    supervised_time = common_time[candidate_indices]
    masks = build_milestone_one_masks(supervised_time)

    selected: list[int] = []
    train_per_year = int(sampling_cfg.get("train_per_year", 0))
    dev_count = int(sampling_cfg.get("dev", 0))
    test_count = int(sampling_cfg.get("test", 0))

    years = supervised_time.year.to_numpy()
    for year in sorted(set(int(year) for year in years[masks.train_mask])):
        year_positions = np.flatnonzero(masks.train_mask & (years == year))
        sampled = _select_evenly_spaced_positions(year_positions.size, train_per_year)
        selected.extend(candidate_indices[year_positions[sampled]].tolist())

    for mask, count in ((masks.dev_mask, dev_count), (masks.test_mask, test_count)):
        positions = np.flatnonzero(mask)
        sampled = _select_evenly_spaced_positions(positions.size, count)
        selected.extend(candidate_indices[positions[sampled]].tolist())

    if not selected:
        return candidate_indices
    return np.array(sorted(set(selected)), dtype=int)


def _add_precip_history_features(
    dataset: xr.Dataset,
    base_precip: xr.DataArray,
    common_time: pd.DatetimeIndex,
    supervised_indices: np.ndarray,
    cadence_hours: int,
) -> xr.Dataset:
    supervised_time = dataset.coords["time"]
    lagged = {}
    for lag in range(4):
        name = "imerg_t" if lag == 0 else f"imerg_t_minus_{lag * cadence_hours}h"
        history_time = common_time[supervised_indices - lag]
        feature = base_precip.sel(time=history_time).assign_coords(time=supervised_time)
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
    max_lag = 3
    sampling_cfg = dict(paths["benchmark"].get("sampling", {}))

    grid_spec = manifest.get(paths["canonical_grid"]["source_key"])
    grid = load_dataarray(grid_spec)
    target_lat = grid.coords["lat"]
    target_lon = grid.coords["lon"]

    timed_specs = [manifest.get("precip_target_raw"), *[manifest.get(key) for key in feature_set.variables]]
    common_time: pd.DatetimeIndex | None = None
    for spec in timed_specs:
        time_index = _load_time_index(spec)
        if time_index is None:
            continue
        common_time = time_index if common_time is None else common_time.intersection(time_index)
    if common_time is None:
        raise RuntimeError("No time-aware datasets were found for feature building")

    supervised_indices = _select_supervised_indices(
        common_time=common_time,
        forecast_horizon_steps=forecast_horizon_steps,
        max_lag=max_lag,
        sampling_cfg=sampling_cfg,
    )
    supervised_time = common_time[supervised_indices]

    predictor_arrays = {}
    for key in feature_set.variables:
        spec = manifest.get(key)
        predictor_arrays[spec.build_name] = load_dataarray(
            spec,
            target_lat=target_lat,
            target_lon=target_lon,
            time_index=supervised_time,
        )
    aligned_predictors = align_dataarrays_on_time(predictor_arrays)
    supervised = xr.Dataset(aligned_predictors)

    flattened_target_indices = sorted(
        {
            *(int(idx) for idx in supervised_indices + forecast_horizon_steps),
            *(int(idx) for lag in range(max_lag + 1) for idx in supervised_indices - lag),
        }
    )
    base_target = load_dataarray(
        manifest.get("precip_target_raw"),
        target_lat=target_lat,
        target_lon=target_lon,
        time_index=common_time[flattened_target_indices],
    )
    target = base_target.sel(time=common_time[supervised_indices + forecast_horizon_steps]).assign_coords(time=supervised.coords["time"])
    supervised["target_precip"] = target
    supervised["target_occurrence"] = (target >= 0.1).astype(np.int8)

    supervised = _add_precip_history_features(
        supervised,
        base_precip=base_target,
        common_time=common_time,
        supervised_indices=supervised_indices,
        cadence_hours=cadence_hours,
    )
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
    supervised.attrs["sampling"] = json.dumps(sampling_cfg, sort_keys=True)
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
