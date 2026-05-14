from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japca.config import load_variables_config


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    variable: str
    role: str
    cadence_hours: int
    grid: str
    units: str
    regrid_to_canonical: bool

    @property
    def build_name(self) -> str:
        if self.key == "precip_target_raw":
            return "target_precip_raw"
        if self.key.endswith("_regrid"):
            return self.key.removesuffix("_regrid")
        return self.key


class DatasetManifest:
    def __init__(self, datasets: dict[str, DatasetSpec], optional_low_res_aerosols: dict[str, DatasetSpec] | None = None):
        self.datasets = datasets
        self.optional_low_res_aerosols = optional_low_res_aerosols or {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "DatasetManifest":
        raw = config or load_variables_config()
        datasets = {
            key: DatasetSpec(
                key=key,
                path=Path(value["path"]),
                variable=value["variable"],
                role=value["role"],
                cadence_hours=int(value["cadence_hours"]),
                grid=value["grid"],
                units=value["units"],
                regrid_to_canonical=bool(value["regrid_to_canonical"]),
            )
            for key, value in raw.get("variables", {}).items()
        }
        optional = {
            key: DatasetSpec(
                key=key,
                path=Path(value["path"]),
                variable=value["variable"],
                role=value["role"],
                cadence_hours=int(value["cadence_hours"]),
                grid=value["grid"],
                units=value["units"],
                regrid_to_canonical=bool(value["regrid_to_canonical"]),
            )
            for key, value in raw.get("optional_low_res_aerosols", {}).items()
        }
        return cls(datasets=datasets, optional_low_res_aerosols=optional)

    def get(self, key: str) -> DatasetSpec:
        if key in self.datasets:
            return self.datasets[key]
        if key in self.optional_low_res_aerosols:
            return self.optional_low_res_aerosols[key]
        raise KeyError(f"Unknown dataset key: {key}")

    def items(self):
        return self.datasets.items()
