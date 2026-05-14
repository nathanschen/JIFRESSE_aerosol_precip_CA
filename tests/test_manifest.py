from pathlib import Path

from japca.data.manifest import DatasetManifest


def test_manifest_resolution_from_config():
    manifest = DatasetManifest.from_config(
        {
            "variables": {
                "pwv_regrid": {
                    "path": "/tmp/pwv.nc",
                    "variable": "pwv",
                    "role": "predictor",
                    "cadence_hours": 6,
                    "grid": "canonical_140x150",
                    "units": "kg_m2",
                    "regrid_to_canonical": False,
                }
            }
        }
    )
    spec = manifest.get("pwv_regrid")
    assert spec.path == Path("/tmp/pwv.nc")
    assert spec.variable == "pwv"
    assert spec.build_name == "pwv"
