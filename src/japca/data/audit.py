from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from japca.config import ensure_directory, load_metrics_config, load_paths_config
from japca.data.alignment import align_dataarrays_on_time, load_dataarray, validate_grid_shape
from japca.data.manifest import DatasetManifest


def run_data_audit() -> dict:
    paths = load_paths_config()
    manifest = DatasetManifest.from_config()
    metrics = load_metrics_config()

    target_spec = manifest.get("precip_target_raw")
    grid_spec = manifest.get(paths["canonical_grid"]["source_key"])
    target_grid = load_dataarray(grid_spec)
    target_lat = target_grid.coords["lat"]
    target_lon = target_grid.coords["lon"]

    arrays = {
        "target_precip_raw": load_dataarray(target_spec, target_lat=target_lat, target_lon=target_lon),
        "pwv": target_grid,
        "topo": load_dataarray(manifest.get("topo_regrid")),
        "totext": load_dataarray(manifest.get("totext")),
    }
    aligned = align_dataarrays_on_time(arrays)
    precip = aligned["target_precip_raw"]

    summary = {
        "canonical_grid": {
            "lat": int(target_lat.size),
            "lon": int(target_lon.size),
        },
        "datasets": {},
        "target_sparsity": {},
    }

    for key, spec in manifest.items():
        arr = load_dataarray(spec, target_lat=target_lat, target_lon=target_lon)
        summary["datasets"][key] = {
            "path": str(spec.path),
            "variable": spec.variable,
            "grid": spec.grid,
            "role": spec.role,
            "units": spec.units,
            "sizes": {name: int(size) for name, size in arr.sizes.items()},
            "grid_valid": validate_grid_shape(arr, int(target_lat.size), int(target_lon.size)),
        }

    precip_np = np.asarray(precip.values)
    summary["target_sparsity"]["rainy_pixel_fraction"] = float((precip_np > 0).mean())
    summary["target_sparsity"]["rainy_timestep_fraction"] = float((precip_np.reshape(precip_np.shape[0], -1) > 0).any(axis=1).mean())
    for threshold in metrics["thresholds_mm_6h"]:
        summary["target_sparsity"][f"frac_ge_{threshold:g}_mm_6h"] = float((precip_np >= threshold).mean())
    return summary


def main() -> None:
    paths = load_paths_config()
    reports_dir = ensure_directory(Path(paths["paths"]["reports_dir"]))
    summary = run_data_audit()
    output_path = reports_dir / "data_audit.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved audit to {output_path}")


if __name__ == "__main__":
    main()
