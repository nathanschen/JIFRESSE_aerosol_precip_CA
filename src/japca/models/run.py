from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader, Subset

from japca.config import ensure_directory, load_metrics_config, load_paths_config
from japca.data.alignment import build_milestone_one_masks
from japca.models.architectures import ConvLSTMUNet, TemporalUNet
from japca.models.training import GridForecastDataset, evaluate_model, save_training_artifacts, train_model


def _load_model_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JAPCA neural model family.")
    parser.add_argument("--model", required=True, choices=["temporal_unet", "convlstm_unet"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit-time", type=int, default=None)
    args = parser.parse_args()

    paths = load_paths_config()
    metrics_cfg = load_metrics_config()
    dataset_path = Path(args.dataset or paths["benchmark"]["default_built_dataset"])
    dataset = xr.open_dataset(dataset_path)
    if args.limit_time:
        dataset = dataset.isel(time=slice(0, args.limit_time))

    config = _load_model_config(args.config)
    source = GridForecastDataset(dataset, model_family=args.model)
    masks = build_milestone_one_masks(dataset.time.values)
    years = dataset["time"].dt.year.values
    train_indices = [idx for idx, year in enumerate(years) if masks.train_mask[idx]]
    dev_indices = [idx for idx, year in enumerate(years) if masks.dev_mask[idx]]
    test_indices = [idx for idx, year in enumerate(years) if masks.test_mask[idx]]

    train_loader = DataLoader(Subset(source, train_indices), batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(Subset(source, dev_indices), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(Subset(source, test_indices), batch_size=config["batch_size"], shuffle=False)

    if args.model == "temporal_unet":
        feature_count = len(source.feature_names)
        model = TemporalUNet(in_channels=feature_count, hidden_channels=int(config["hidden_channels"]))
    else:
        input_channels = 1 + len(source.static_features)
        model = ConvLSTMUNet(input_channels=input_channels, hidden_channels=int(config["convlstm_hidden_channels"]))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
    )
    dev_metrics = evaluate_model(model, dev_loader, dataset.sel(time=masks.dev_mask), metrics_cfg, device)
    test_metrics = evaluate_model(model, test_loader, dataset.sel(time=masks.test_mask), metrics_cfg, device)
    output_dir = ensure_directory(Path(paths["paths"]["models_dir"]) / args.model)
    checkpoint_path = save_training_artifacts(output_dir, {"dev": dev_metrics, "test": test_metrics, "config": config}, model)
    print(json.dumps({"dev": dev_metrics, "test": test_metrics, "checkpoint": str(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()
