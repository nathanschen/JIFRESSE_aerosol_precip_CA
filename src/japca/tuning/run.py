from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from japca.baselines.hurdle import HurdleXGB
from japca.baselines.run import DERIVED_COLUMN_MAP
from japca.config import ensure_directory, load_metrics_config, load_paths_config
from japca.data.alignment import build_milestone_one_masks
from japca.features.registry import resolve_feature_set
from japca.metrics.weather import evaluate_forecast


def _load_tuning_config(study: str) -> dict:
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "tuning"
    candidate = config_dir / f"{study}.yaml"
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    raise FileNotFoundError(f"No tuning config found for study {study}")


def _suggest(trial, name: str, spec: dict):
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search space type for {name}: {spec}")


def _resolve_feature_columns(dataset: xr.Dataset, tuning_cfg: dict) -> list[str]:
    available = [name for name in dataset.data_vars if name not in {"target_precip", "target_occurrence"}]
    feature_names = list(available)

    feature_set_name = tuning_cfg.get("feature_set")
    if feature_set_name:
        selected: list[str] = []
        available_set = set(available)
        feature_set = resolve_feature_set(feature_set_name)
        for membership in feature_set.group_membership.values():
            for variable in membership["variables"]:
                build_name = "target_precip_raw" if variable == "precip_target_raw" else variable.removesuffix("_regrid")
                if build_name in available_set:
                    selected.append(build_name)
            for derived_name in membership["derived"]:
                selected.extend(column for column in DERIVED_COLUMN_MAP.get(derived_name, ()) if column in available_set)
        selected_set = set(selected)
        feature_names = [name for name in available if name in selected_set]

    extra_feature_columns = tuning_cfg.get("extra_feature_columns", [])
    for name in extra_feature_columns:
        if name in available and name not in feature_names:
            feature_names.append(name)

    excluded = set(tuning_cfg.get("drop_feature_columns", []))
    feature_names = [name for name in feature_names if name not in excluded]
    if not feature_names:
        raise ValueError("Resolved feature column list is empty for the tuning study.")
    return feature_names


def _flatten(dataset: xr.Dataset, feature_names: list[str]):
    x_values = np.stack([np.asarray(dataset[name].values).reshape(dataset.sizes["time"], -1) for name in feature_names], axis=-1).reshape(-1, len(feature_names))
    y_amount = np.asarray(dataset["target_precip"].values).reshape(-1)
    return x_values, y_amount


def _split_hurdle_params(params: dict) -> tuple[dict, dict, float | None]:
    threshold = params.get("occurrence_probability_threshold")
    model_params = {key: value for key, value in params.items() if key != "occurrence_probability_threshold"}
    classifier_prefixed = {key.removeprefix("classifier__"): value for key, value in model_params.items() if key.startswith("classifier__")}
    regressor_prefixed = {key.removeprefix("regressor__"): value for key, value in model_params.items() if key.startswith("regressor__")}
    if classifier_prefixed or regressor_prefixed:
        return classifier_prefixed, regressor_prefixed, threshold
    return dict(model_params), dict(model_params), threshold


def _predict_tabular_amount(
    pipeline: str,
    model: HurdleXGB,
    x_eval: np.ndarray,
    eval_ds: xr.Dataset,
) -> np.ndarray:
    if pipeline == "persistence_hybrid":
        return np.asarray(eval_ds["imerg_t"].values)
    if pipeline == "persistence_hybrid_gated":
        pred_prob = model.predict_occurrence_probability(x_eval).reshape(eval_ds["target_occurrence"].shape)
        return np.where(pred_prob >= model.occurrence_probability_threshold, np.asarray(eval_ds["imerg_t"].values), 0.0)
    return model.predict_amount(x_eval).reshape(eval_ds["target_precip"].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JAPCA Optuna tuning studies.")
    parser.add_argument("--model", required=True, choices=["tabular", "temporal_unet", "convlstm_unet"])
    parser.add_argument("--study", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised in real runtime only
        raise RuntimeError("Optuna is required for tuning. Install japca with optuna available.") from exc

    paths = load_paths_config()
    metrics_cfg = load_metrics_config()
    tuning_cfg = _load_tuning_config(args.study)
    dataset_path = Path(args.dataset or paths["benchmark"]["default_built_dataset"])
    dataset = xr.open_dataset(dataset_path)
    masks = build_milestone_one_masks(dataset.time.values)

    train_ds = dataset.sel(time=masks.train_mask)
    dev_ds = dataset.sel(time=masks.dev_mask)
    feature_names = _resolve_feature_columns(dataset, tuning_cfg)
    x_train, y_train_amount = _flatten(train_ds, feature_names)
    x_dev, _ = _flatten(dev_ds, feature_names)

    def objective(trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in tuning_cfg["search_space"].items()}
        if args.model != "tabular":
            return 0.0
        classifier_params, regressor_params, tuned_threshold = _split_hurdle_params(params)
        threshold = (
            float(tuned_threshold)
            if tuned_threshold is not None
            else float(metrics_cfg["metric_defaults"]["occurrence_probability_threshold"])
        )
        model = HurdleXGB(
            rainy_threshold=metrics_cfg["metric_defaults"]["rainy_pixel_threshold_mm_6h"],
            occurrence_probability_threshold=threshold,
            classifier_params=classifier_params,
            regressor_params=regressor_params,
        )
        model.fit(x_train, y_train_amount)
        pipeline = tuning_cfg.get("pipeline", "plain_hurdle")
        pred_amount = _predict_tabular_amount(pipeline, model, x_dev, dev_ds)
        pred_prob = model.predict_occurrence_probability(x_dev).reshape(dev_ds["target_occurrence"].shape)
        metrics = evaluate_forecast(
            y_true_amount=np.asarray(dev_ds["target_precip"].values),
            y_pred_amount=pred_amount,
            y_occurrence_prob=pred_prob,
            thresholds=metrics_cfg["thresholds_mm_6h"],
            fss_windows=metrics_cfg["fss_windows"],
            ranking_weights=metrics_cfg["ranking_weights"],
            reliability_bin_edges=metrics_cfg["reliability_bins"],
        )
        trial.set_user_attr("metrics", metrics)
        return float(metrics["composite_score"])

    studies_dir = ensure_directory(Path(paths["paths"]["studies_dir"]))
    storage = tuning_cfg["storage"].replace("outputs/studies", str(studies_dir))
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    for params in tuning_cfg.get("enqueue_trials", []):
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=int(tuning_cfg["trials"]))
    summary = {
        "study": args.study,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "pipeline": tuning_cfg.get("pipeline", "plain_hurdle"),
        "feature_columns": feature_names,
        "feature_set": tuning_cfg.get("feature_set"),
        "best_metrics": study.best_trial.user_attrs.get("metrics", {}),
    }
    (studies_dir / f"{args.study}_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
