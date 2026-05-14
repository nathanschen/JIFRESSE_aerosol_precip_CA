from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from japca.metrics.weather import evaluate_forecast


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1) * error).mean()


class GridForecastDataset(Dataset):
    def __init__(self, dataset: xr.Dataset, model_family: str) -> None:
        self.dataset = dataset
        self.model_family = model_family
        self.feature_names = [name for name in dataset.data_vars if name not in {"target_precip", "target_occurrence"}]
        self.sequence_features = [name for name in self.feature_names if name.startswith("imerg_t")]
        self.static_features = [name for name in self.feature_names if name not in self.sequence_features]

    def __len__(self) -> int:
        return self.dataset.sizes["time"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occurrence = torch.from_numpy(np.asarray(self.dataset["target_occurrence"].isel(time=index).values)).float().unsqueeze(0)
        amount = torch.from_numpy(np.asarray(self.dataset["target_precip"].isel(time=index).values)).float().unsqueeze(0)
        if self.model_family == "convlstm_unet":
            ordered = sorted(self.sequence_features, key=lambda name: (0 if name == "imerg_t" else int(name.split("_minus_")[1].removesuffix("h"))))
            ordered = list(reversed(ordered))
            frames = []
            for feature in ordered:
                dynamic = np.asarray(self.dataset[feature].isel(time=index).values)[None, ...]
                static = [np.asarray(self.dataset[name].isel(time=index).values) for name in self.static_features]
                frame = np.concatenate([dynamic, *[arr[None, ...] for arr in static]], axis=0)
                frames.append(frame)
            inputs = torch.from_numpy(np.stack(frames, axis=0)).float()
        else:
            channels = [np.asarray(self.dataset[name].isel(time=index).values) for name in self.feature_names]
            inputs = torch.from_numpy(np.stack(channels, axis=0)).float()
        return inputs, occurrence, amount


@dataclass
class TrainResult:
    metrics: dict
    checkpoint_path: Path | None = None


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_occ, batch_amount in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_occ = batch_occ.to(device)
            batch_amount = batch_amount.to(device)
            optimizer.zero_grad()
            occ_logits, amount_pred = model(batch_inputs)
            occ_loss = focal_bce_with_logits(occ_logits, batch_occ)
            rain_mask = batch_occ > 0.5
            transformed_target = torch.log1p(batch_amount)
            if rain_mask.any():
                amount_loss = quantile_loss(amount_pred[rain_mask], transformed_target[rain_mask], quantile=0.5)
            else:
                amount_loss = torch.tensor(0.0, device=device)
            heavy_mask = batch_amount >= 5.0
            heavy_penalty = torch.tensor(0.0, device=device)
            if heavy_mask.any():
                heavy_penalty = nn.functional.l1_loss(amount_pred[heavy_mask], transformed_target[heavy_mask])
            loss = occ_loss + amount_loss + 0.2 * heavy_penalty
            loss.backward()
            optimizer.step()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    source_dataset: xr.Dataset,
    metrics_cfg: dict,
    device: torch.device,
) -> dict:
    model.eval()
    amount_predictions = []
    prob_predictions = []
    with torch.no_grad():
        for batch_inputs, _, _ in loader:
            batch_inputs = batch_inputs.to(device)
            occ_logits, amount_pred = model(batch_inputs)
            prob = torch.sigmoid(occ_logits).cpu().numpy()
            amount = np.expm1(amount_pred.cpu().numpy())
            prob_predictions.append(prob)
            amount_predictions.append(np.where(prob >= metrics_cfg["metric_defaults"]["occurrence_probability_threshold"], amount, 0.0))
    pred_amount = np.concatenate(amount_predictions, axis=0)[:, 0]
    pred_prob = np.concatenate(prob_predictions, axis=0)[:, 0]
    return evaluate_forecast(
        y_true_amount=np.asarray(source_dataset["target_precip"].values),
        y_pred_amount=pred_amount,
        y_occurrence_prob=pred_prob,
        thresholds=metrics_cfg["thresholds_mm_6h"],
        fss_windows=metrics_cfg["fss_windows"],
        ranking_weights=metrics_cfg["ranking_weights"],
        reliability_bin_edges=metrics_cfg["reliability_bins"],
    )


def save_training_artifacts(output_dir: Path, metrics: dict, model: nn.Module) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return checkpoint_path
