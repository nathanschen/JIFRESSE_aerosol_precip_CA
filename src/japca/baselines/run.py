from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from japca.baselines.hurdle import HurdleXGB
from japca.config import ensure_directory, load_metrics_config, load_paths_config
from japca.data.alignment import build_milestone_one_masks
from japca.features.registry import resolve_feature_set
from japca.metrics.weather import evaluate_forecast


DERIVED_COLUMN_MAP = {
    "precip_history_lags": ("imerg_t", "imerg_t_minus_6h", "imerg_t_minus_12h", "imerg_t_minus_18h"),
    "precip_lag_differences": ("imerg_diff_t_minus_6h", "imerg_diff_6h_minus_12h"),
    "precip_rolling_extrema": ("imerg_rolling_max_24h", "imerg_rolling_min_24h"),
    "low_level_moisture_flux": ("moisture_flux_u10", "moisture_flux_v10"),
    "mid_level_moisture_flux": ("moisture_flux_u700", "moisture_flux_v700"),
    "wind_speed": ("wind_speed_10m", "wind_speed_700hpa"),
    "vertical_shear": ("vertical_shear_u", "vertical_shear_v", "vertical_shear_speed"),
    "terrain_gradients": ("terrain_slope_lat", "terrain_slope_lon", "terrain_slope_magnitude", "terrain_aspect"),
    "upslope_flow_proxy": ("upslope_flow_10m", "upslope_flow_700hpa"),
    "coastal_distance_proxy": ("coastal_distance_proxy",),
}


def _feature_columns(dataset: xr.Dataset) -> list[str]:
    return [name for name in dataset.data_vars if name not in {"target_precip", "target_occurrence"}]


def _to_tabular(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_names = _feature_columns(dataset)
    stacked = []
    for name in feature_names:
        stacked.append(np.asarray(dataset[name].values).reshape(dataset.sizes["time"], -1))
    x_values = np.stack(stacked, axis=-1).reshape(-1, len(feature_names))
    y_amount = np.asarray(dataset["target_precip"].values).reshape(-1)
    y_occ = np.asarray(dataset["target_occurrence"].values).reshape(-1)
    years = np.repeat(dataset["time"].dt.year.values, dataset.sizes["lat"] * dataset.sizes["lon"])
    return x_values, y_amount, y_occ, years


def _reshape_map(flat: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
    return flat.reshape(dataset.sizes["time"], dataset.sizes["lat"], dataset.sizes["lon"])


def _run_climatology(train_ds: xr.Dataset, eval_ds: xr.Dataset, metrics_cfg: dict) -> dict:
    mean_amount = np.asarray(train_ds["target_precip"].mean(dim="time").values)
    mean_prob = np.asarray(train_ds["target_occurrence"].mean(dim="time").values)
    pred_amount = np.broadcast_to(mean_amount, eval_ds["target_precip"].shape)
    pred_prob = np.broadcast_to(mean_prob, eval_ds["target_occurrence"].shape)
    return evaluate_forecast(
        y_true_amount=np.asarray(eval_ds["target_precip"].values),
        y_pred_amount=pred_amount,
        y_occurrence_prob=pred_prob,
        thresholds=metrics_cfg["thresholds_mm_6h"],
        fss_windows=metrics_cfg["fss_windows"],
        ranking_weights=metrics_cfg["ranking_weights"],
        reliability_bin_edges=metrics_cfg["reliability_bins"],
    )


def _run_persistence(eval_ds: xr.Dataset, metrics_cfg: dict) -> dict:
    if "imerg_t" not in eval_ds:
        raise KeyError("Built dataset is missing imerg_t persistence feature")
    pred_amount = np.asarray(eval_ds["imerg_t"].values)
    pred_prob = (pred_amount >= metrics_cfg["metric_defaults"]["rainy_pixel_threshold_mm_6h"]).astype(float)
    return evaluate_forecast(
        y_true_amount=np.asarray(eval_ds["target_precip"].values),
        y_pred_amount=pred_amount,
        y_occurrence_prob=pred_prob,
        thresholds=metrics_cfg["thresholds_mm_6h"],
        fss_windows=metrics_cfg["fss_windows"],
        ranking_weights=metrics_cfg["ranking_weights"],
        reliability_bin_edges=metrics_cfg["reliability_bins"],
    )


def _run_hurdle(train_ds: xr.Dataset, eval_ds: xr.Dataset, metrics_cfg: dict) -> dict:
    x_train, y_amount_train, _, _ = _to_tabular(train_ds)
    x_eval, _, _, _ = _to_tabular(eval_ds)
    model = HurdleXGB(
        rainy_threshold=metrics_cfg["metric_defaults"]["rainy_pixel_threshold_mm_6h"],
        occurrence_probability_threshold=metrics_cfg["metric_defaults"]["occurrence_probability_threshold"],
    )
    model.fit(x_train, y_amount_train)
    pred_amount = model.predict_amount(x_eval)
    pred_prob = model.predict_occurrence_probability(x_eval)
    return evaluate_forecast(
        y_true_amount=np.asarray(eval_ds["target_precip"].values),
        y_pred_amount=_reshape_map(pred_amount, eval_ds),
        y_occurrence_prob=_reshape_map(pred_prob, eval_ds),
        thresholds=metrics_cfg["thresholds_mm_6h"],
        fss_windows=metrics_cfg["fss_windows"],
        ranking_weights=metrics_cfg["ranking_weights"],
        reliability_bin_edges=metrics_cfg["reliability_bins"],
    )


def _group_to_columns(dataset: xr.Dataset, feature_set_name: str) -> dict[str, list[str]]:
    feature_names = set(_feature_columns(dataset))
    feature_set = resolve_feature_set(feature_set_name)
    result: dict[str, list[str]] = {}
    for group_name, membership in feature_set.group_membership.items():
        columns: list[str] = []
        for variable in membership["variables"]:
            build_name = "target_precip_raw" if variable == "precip_target_raw" else variable.removesuffix("_regrid")
            if build_name in feature_names:
                columns.append(build_name)
        for derived_name in membership["derived"]:
            columns.extend(column for column in DERIVED_COLUMN_MAP.get(derived_name, ()) if column in feature_names)
        result[group_name] = sorted(set(columns))
    return result


def _metric_anchor(metrics: dict) -> float:
    threshold = metrics["threshold_metrics"]["1"]["csi"] if "1" in metrics["threshold_metrics"] else 0.0
    return float(metrics["composite_score"] + threshold)


def _grouped_permutation_importance(dataset: xr.Dataset, feature_set_name: str, metrics_cfg: dict) -> list[dict]:
    x_values, y_amount, _, years = _to_tabular(dataset)
    groups = _group_to_columns(dataset, feature_set_name)
    feature_names = _feature_columns(dataset)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    importances = []

    screening_years = [year for year in sorted(set(years.tolist())) if year <= 2022]
    for holdout_year in screening_years:
        train_mask = (years <= 2022) & (years != holdout_year)
        holdout_mask = years == holdout_year
        if train_mask.sum() == 0 or holdout_mask.sum() == 0:
            continue
        model = HurdleXGB(
            rainy_threshold=metrics_cfg["metric_defaults"]["rainy_pixel_threshold_mm_6h"],
            occurrence_probability_threshold=metrics_cfg["metric_defaults"]["occurrence_probability_threshold"],
        )
        model.fit(x_values[train_mask], y_amount[train_mask])
        base_amount = model.predict_amount(x_values[holdout_mask])
        base_prob = model.predict_occurrence_probability(x_values[holdout_mask])

        holdout_ds = dataset.sel(time=dataset["time"].dt.year == holdout_year)
        base_metrics = evaluate_forecast(
            y_true_amount=np.asarray(holdout_ds["target_precip"].values),
            y_pred_amount=_reshape_map(base_amount, holdout_ds),
            y_occurrence_prob=_reshape_map(base_prob, holdout_ds),
            thresholds=metrics_cfg["thresholds_mm_6h"],
            fss_windows=metrics_cfg["fss_windows"],
            ranking_weights=metrics_cfg["ranking_weights"],
            reliability_bin_edges=metrics_cfg["reliability_bins"],
        )
        base_score = _metric_anchor(base_metrics)

        eval_frame = x_values[holdout_mask].copy()
        sample = min(eval_frame.shape[0], 5000)
        rng = np.random.default_rng(42 + int(holdout_year))
        selection = rng.choice(eval_frame.shape[0], size=sample, replace=False)
        eval_sample = eval_frame[selection]
        truth_sample = y_amount[holdout_mask][selection]
        truth_shape = (sample, 1, 1)
        for group_name, columns in groups.items():
            if not columns:
                continue
            permuted = eval_sample.copy()
            for column in columns:
                idx = feature_index[column]
                permuted[:, idx] = rng.permutation(permuted[:, idx])
            perm_amount = model.predict_amount(permuted).reshape(truth_shape)
            perm_prob = model.predict_occurrence_probability(permuted).reshape(truth_shape)
            perm_metrics = evaluate_forecast(
                y_true_amount=truth_sample.reshape(truth_shape),
                y_pred_amount=perm_amount,
                y_occurrence_prob=perm_prob,
                thresholds=metrics_cfg["thresholds_mm_6h"],
                fss_windows=metrics_cfg["fss_windows"],
                ranking_weights=metrics_cfg["ranking_weights"],
                reliability_bin_edges=metrics_cfg["reliability_bins"],
            )
            importances.append(
                {
                    "holdout_year": int(holdout_year),
                    "group": group_name,
                    "baseline_score": base_score,
                    "permuted_score": _metric_anchor(perm_metrics),
                    "importance_delta": base_score - _metric_anchor(perm_metrics),
                }
            )
    return importances


def _maybe_compute_shap(dataset: xr.Dataset, metrics_cfg: dict) -> dict:
    shap_module = importlib.util.find_spec("shap")
    if shap_module is None:
        return {"status": "missing_dependency", "detail": "Install shap to enable SHAP summaries."}
    shap = importlib.import_module("shap")
    masks = build_milestone_one_masks(dataset.time.values)
    train_ds = dataset.sel(time=masks.train_mask)
    dev_ds = dataset.sel(time=masks.dev_mask)
    x_train, y_amount_train, _, _ = _to_tabular(train_ds)
    x_dev, _, _, _ = _to_tabular(dev_ds)
    model = HurdleXGB(
        rainy_threshold=metrics_cfg["metric_defaults"]["rainy_pixel_threshold_mm_6h"],
        occurrence_probability_threshold=metrics_cfg["metric_defaults"]["occurrence_probability_threshold"],
    )
    model.fit(x_train, y_amount_train)
    sample = min(x_dev.shape[0], 1000)
    explainer = shap.Explainer(model.classifier, x_train[:sample])
    values = explainer(x_dev[:sample])
    means = np.abs(values.values).mean(axis=0)
    return {
        "status": "ok",
        "sample_size": sample,
        "mean_abs_shap": {name: float(score) for name, score in zip(_feature_columns(dataset), means)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JAPCA tabular baselines.")
    parser.add_argument("--suite", required=True, help="Benchmark suite name")
    parser.add_argument("--dataset", default=None, help="Optional built dataset path")
    parser.add_argument("--limit-time", type=int, default=None, help="Optional number of timesteps to subset for fast smoke runs")
    args = parser.parse_args()

    paths = load_paths_config()
    metrics_cfg = load_metrics_config()
    feature_set_name = paths["benchmark"]["default_feature_set"]
    dataset_path = Path(args.dataset or paths["benchmark"]["default_built_dataset"])
    dataset = xr.open_dataset(dataset_path)
    if args.limit_time:
        dataset = dataset.isel(time=slice(0, args.limit_time))

    split_masks = build_milestone_one_masks(dataset.time.values)
    train_ds = dataset.sel(time=split_masks.train_mask)
    dev_ds = dataset.sel(time=split_masks.dev_mask)
    test_ds = dataset.sel(time=split_masks.test_mask)

    suite_dir = ensure_directory(Path(paths["paths"]["baselines_dir"]) / args.suite)
    results = {
        "suite": args.suite,
        "dataset": str(dataset_path),
        "models": {
            "climatology_dev": _run_climatology(train_ds, dev_ds, metrics_cfg),
            "climatology_test": _run_climatology(train_ds, test_ds, metrics_cfg),
            "persistence_dev": _run_persistence(dev_ds, metrics_cfg),
            "persistence_test": _run_persistence(test_ds, metrics_cfg),
            "hurdle_xgb_dev": _run_hurdle(train_ds, dev_ds, metrics_cfg),
            "hurdle_xgb_test": _run_hurdle(train_ds, test_ds, metrics_cfg),
        },
    }
    results["feature_group_importance"] = _grouped_permutation_importance(dataset, feature_set_name, metrics_cfg)
    results["shap_summary"] = _maybe_compute_shap(dataset, metrics_cfg)

    (suite_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (suite_dir / "feature_group_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["holdout_year", "group", "baseline_score", "permuted_score", "importance_delta"])
        writer.writeheader()
        for row in results["feature_group_importance"]:
            writer.writerow(row)
    print(json.dumps(results, indent=2))
    print(f"Saved baseline results to {suite_dir}")


if __name__ == "__main__":
    main()
