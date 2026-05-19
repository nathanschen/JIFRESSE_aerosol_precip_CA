from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from japca.data.manifest import DatasetSpec


def _to_datetime_index(time_values: Iterable[object]) -> pd.DatetimeIndex:
    try:
        return pd.DatetimeIndex(pd.to_datetime(list(time_values)))
    except (TypeError, ValueError):
        converted = []
        for value in time_values:
            converted.append(
                pd.Timestamp(
                    year=int(value.year),
                    month=int(value.month),
                    day=int(value.day),
                    hour=int(getattr(value, "hour", 0)),
                    minute=int(getattr(value, "minute", 0)),
                    second=int(getattr(value, "second", 0)),
                )
            )
        return pd.DatetimeIndex(converted)


def canonicalize_coords(data_array: xr.DataArray) -> xr.DataArray:
    da = data_array
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    if "time" in da.coords:
        da = da.assign_coords(time=_to_datetime_index(da["time"].values))
    ordered_dims = [dim for dim in ("time", "lat", "lon") if dim in da.dims]
    remaining_dims = [dim for dim in da.dims if dim not in ordered_dims]
    if ordered_dims:
        da = da.transpose(*ordered_dims, *remaining_dims)
    return da


def load_dataarray(
    spec: DatasetSpec,
    target_lat: xr.DataArray | None = None,
    target_lon: xr.DataArray | None = None,
    time_index: pd.DatetimeIndex | None = None,
) -> xr.DataArray:
    dataset = xr.open_dataset(spec.path)
    if spec.variable not in dataset:
        raise KeyError(f"{spec.variable} not found in {spec.path}")
    da = canonicalize_coords(dataset[spec.variable])
    if time_index is not None and "time" in da.dims:
        da = da.sel(time=time_index)
    if spec.regrid_to_canonical and target_lat is not None and target_lon is not None:
        da = da.interp(lat=target_lat, lon=target_lon)
    if np.issubdtype(da.dtype, np.floating):
        da = da.astype(np.float32)
    return da.rename(spec.build_name)


def align_dataarrays_on_time(arrays: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    timed = {name: arr for name, arr in arrays.items() if "time" in arr.dims}
    static = {name: arr for name, arr in arrays.items() if "time" not in arr.dims}
    if timed:
        aligned = xr.align(*timed.values(), join="inner")
        timed = {name: arr for name, arr in zip(timed.keys(), aligned)}
        time_index = next(iter(timed.values())).coords["time"]
        for name, arr in static.items():
            expanded = arr.expand_dims(time=time_index).transpose("time", "lat", "lon")
            static[name] = expanded.assign_coords(time=time_index)
    return {**timed, **static}


def validate_grid_shape(data_array: xr.DataArray, expected_lat: int, expected_lon: int) -> bool:
    if "lat" not in data_array.dims or "lon" not in data_array.dims:
        return False
    return int(data_array.sizes["lat"]) == expected_lat and int(data_array.sizes["lon"]) == expected_lon


def valid_time_mask(dataset: xr.Dataset) -> xr.DataArray:
    mask = xr.DataArray(np.ones(dataset.sizes["time"], dtype=bool), dims=("time",), coords={"time": dataset.time})
    for name, variable in dataset.data_vars.items():
        if "time" not in variable.dims:
            continue
        reduce_dims = [dim for dim in variable.dims if dim != "time"]
        nulls = variable.isnull()
        if reduce_dims:
            nulls = nulls.any(dim=reduce_dims)
        mask = mask & (~nulls)
    return mask


@dataclass(frozen=True)
class MilestoneOneMasks:
    screening_years: tuple[int, ...]
    train_mask: np.ndarray
    dev_mask: np.ndarray
    test_mask: np.ndarray


def build_milestone_one_masks(time_values: Iterable[pd.Timestamp]) -> MilestoneOneMasks:
    years = _to_datetime_index(time_values).year.to_numpy()
    screening = tuple(sorted(set(int(year) for year in years if 2017 <= int(year) <= 2022)))
    return MilestoneOneMasks(
        screening_years=screening,
        train_mask=((years >= 2017) & (years <= 2021)),
        dev_mask=(years == 2022),
        test_mask=(years == 2023),
    )
