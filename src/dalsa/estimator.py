"""Difficulty estimator with explicit P(Hard | x) semantics."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


TARGET_MODES = {"hard_bce", "soft_fail_regression", "value_regression"}


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 512)
        self.block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.head(x).squeeze(-1)


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


def build_estimator_model(hidden_dim: int, network: str = "residual_mlp_v3") -> nn.Module:
    if network in {"residual_mlp_v3", "mlp", "value_network"}:
        return ValueNetwork(hidden_dim)
    if network in {"linear", "linear_probe"}:
        return LinearProbe(hidden_dim)
    raise ValueError(f"Unsupported difficulty estimator network: {network}")


class DifficultyEstimator:
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        device: str = "cuda",
        target_mode: str = "hard_bce",
    ):
        if target_mode not in TARGET_MODES:
            raise ValueError(f"Unsupported estimator target mode: {target_mode}")
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.target_mode = target_mode

    def _hard_scores_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(logits)
        if self.target_mode == "value_regression":
            return 1.0 - scores
        return scores

    @torch.no_grad()
    def predict_proba_hard(self, h0: torch.Tensor) -> float:
        h = h0.to(self.device)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logit = self.model(h).float()
        return float(self._hard_scores_from_logits(logit).item())

    @torch.no_grad()
    def predict_batch_proba_hard(self, h0: torch.Tensor) -> np.ndarray:
        logits = self.model(h0.to(self.device)).float()
        return self._hard_scores_from_logits(logits).cpu().numpy()

    def is_hard(self, h0: torch.Tensor) -> bool:
        return self.predict_proba_hard(h0) >= self.threshold

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "DifficultyEstimator":
        path = Path(path)
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("positive_label") != "hard":
            raise ValueError(f"Estimator at {path} is not a P(Hard) checkpoint.")
        model = build_estimator_model(
            int(config["hidden_dim"]),
            network=str(config.get("network", "residual_mlp_v3")),
        )
        ckpt = torch.load(path / "checkpoint.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return cls(
            model=model,
            threshold=float(config.get("threshold", 0.5)),
            device=device,
            target_mode=str(config.get("target_mode", "hard_bce")),
        )


def infer_num_rollouts(data: dict[str, Any], num_rollouts: int | None = None) -> int:
    metadata = data.get("metadata", {})
    counts = [metadata[key] for key in ("num_rollouts", "rollout_count", "K") if key in metadata]
    counts.extend(item["num_rollouts"] for item in data["items"] if "num_rollouts" in item)
    if num_rollouts is not None:
        counts.append(num_rollouts)
    if not counts:
        raise ValueError("Missing num_rollouts metadata; supply the actual K with --num-rollouts for legacy H0 data.")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or k <= 0 for k in counts):
        raise ValueError("num_rollouts must be a positive integer.")
    if len(set(counts)) != 1:
        raise ValueError("Conflicting num_rollouts values in metadata, items, or --num-rollouts.")
    return int(counts[0])


def soft_fail_targets(data: dict[str, Any]) -> torch.Tensor:
    k = infer_num_rollouts(data)
    values = []
    for item in data["items"]:
        correct_count = item.get("correct_count")
        if isinstance(correct_count, bool) or not isinstance(correct_count, Integral) or not 0 <= correct_count <= k:
            raise ValueError(f"correct_count must be an integer between 0 and num_rollouts={k}.")
        values.append(1.0 - (correct_count / k))
    return torch.tensor(values, dtype=torch.float32)


def value_targets(data: dict[str, Any]) -> torch.Tensor:
    return 1.0 - soft_fail_targets(data)


def hard_targets(data: dict[str, Any]) -> torch.Tensor:
    return torch.tensor([1.0 if item["is_hard"] else 0.0 for item in data["items"]], dtype=torch.float32)


def tensors_from_h0_dataset(
    data: dict[str, Any],
    target_mode: str = "hard_bce",
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported estimator target mode: {target_mode}")
    items = data["items"]
    x = torch.stack([item["h0"].float() for item in items])
    if target_mode == "soft_fail_regression":
        y = soft_fail_targets(data)
    elif target_mode == "value_regression":
        y = value_targets(data)
    else:
        y = hard_targets(data)
    return x, y


def hard_scores_from_logits(logits: torch.Tensor, target_mode: str) -> torch.Tensor:
    scores = torch.sigmoid(logits)
    if target_mode == "value_regression":
        return 1.0 - scores
    return scores


def train_estimator(
    train_data: dict[str, Any],
    val_data: dict[str, Any] | None,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    threshold: float | None = None,
    network: str = "residual_mlp_v3",
    target_mode: str = "hard_bce",
    device: str = "cuda",
) -> dict[str, Any]:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported estimator target mode: {target_mode}")
    output_dir = Path(output_dir)
    soft_fail_targets(train_data)
    if val_data is not None:
        soft_fail_targets(val_data)
    x_train, y_train = tensors_from_h0_dataset(train_data, target_mode=target_mode)
    hidden_dim = int(train_data["hidden_dim"])
    model = build_estimator_model(hidden_dim, network=network).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    hard_train = hard_targets(train_data)
    n_hard = float(hard_train.sum().item())
    n_easy = float(len(hard_train) - n_hard)
    weight_hard = len(y_train) / (2.0 * n_hard) if n_hard else 1.0
    weight_easy = len(y_train) / (2.0 * n_easy) if n_easy else 1.0

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).float()
            if target_mode == "hard_bce":
                weights = torch.where(
                    batch_y > 0.5,
                    torch.tensor(weight_hard, device=device),
                    torch.tensor(weight_easy, device=device),
                )
                loss = F.binary_cross_entropy_with_logits(logits, batch_y, weight=weights)
            else:
                scores = torch.sigmoid(logits)
                loss = F.mse_loss(scores, batch_y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    if val_data is not None and threshold is None:
        threshold = find_best_threshold(model, val_data, device, target_mode=target_mode)
    if threshold is None:
        threshold = 0.5

    metrics = evaluate_estimator(model, val_data or train_data, threshold, device, target_mode=target_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "history": history}, output_dir / "checkpoint.pt")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "hidden_dim": hidden_dim,
                "num_rollouts": infer_num_rollouts(train_data),
                "threshold": threshold,
                "positive_label": "hard",
                "target_mode": target_mode,
                "output_semantics": (
                    "1 - sigmoid(logit) == P(Hard | x)"
                    if target_mode == "value_regression"
                    else "sigmoid(logit) == P(Hard | x)"
                ),
                "network": network,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    return metrics


@torch.no_grad()
def find_best_threshold(model: nn.Module, data: dict[str, Any], device: str, target_mode: str = "hard_bce") -> float:
    model.eval()
    x, _ = tensors_from_h0_dataset(data, target_mode=target_mode)
    probs = hard_scores_from_logits(model(x.to(device)).float(), target_mode).cpu().numpy()
    labels = hard_targets(data).numpy().astype(int)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


@torch.no_grad()
def evaluate_estimator(
    model: nn.Module,
    data: dict[str, Any],
    threshold: float,
    device: str,
    target_mode: str = "hard_bce",
) -> dict[str, float]:
    model.eval()
    x, y = tensors_from_h0_dataset(data, target_mode=target_mode)
    labels = hard_targets(data).numpy().astype(int)
    logits = model(x.to(device)).float()
    probs = hard_scores_from_logits(logits, target_mode).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    soft_fail = soft_fail_targets(data).numpy()
    values = 1.0 - soft_fail
    pred_values = 1.0 - probs
    metrics = {
        "threshold": float(threshold),
        "num_items": int(len(labels)),
        "hard_count": int(labels.sum()),
        "target_mean": float(y.numpy().mean()),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "hard_rate": float(labels.mean()),
        "predicted_hard_rate": float(preds.mean()),
        "mean_p_hard": float(probs.mean()),
        "soft_fail_mse": float(mean_squared_error(soft_fail, probs)),
        "soft_fail_mae": float(mean_absolute_error(soft_fail, probs)),
        "value_mse": float(mean_squared_error(values, pred_values)),
        "value_mae": float(mean_absolute_error(values, pred_values)),
    }
    if target_mode == "value_regression":
        metrics["target_mse"] = metrics["value_mse"]
        metrics["target_mae"] = metrics["value_mae"]
    elif target_mode == "soft_fail_regression":
        metrics["target_mse"] = metrics["soft_fail_mse"]
        metrics["target_mae"] = metrics["soft_fail_mae"]
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics
