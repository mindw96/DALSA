#!/usr/bin/env python
"""Train a P(Hard | x) difficulty estimator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dalsa.estimator import infer_num_rollouts, train_estimator
from dalsa.modeling import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--network",
        choices=["residual_mlp_v3", "linear"],
        default="residual_mlp_v3",
    )
    parser.add_argument(
        "--target-mode",
        choices=["hard_bce", "soft_fail_regression", "value_regression"],
        default="hard_bce",
        help="Estimator supervision target. All modes expose a P(Hard|x)-style score at inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rollouts", type=int, default=None,
                        help="Actual K for legacy H0 files missing rollout metadata; must agree with any existing K.")
    args = parser.parse_args()

    set_seed(args.seed)
    train_data = torch.load(args.train, map_location="cpu")
    val_data = torch.load(args.val, map_location="cpu") if args.val else None
    for data in (train_data, val_data):
        if data is not None:
            data["metadata"] = {**data.get("metadata", {}),
                                "num_rollouts": infer_num_rollouts(data, args.num_rollouts)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = train_estimator(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold=args.threshold,
        network=args.network,
        target_mode=args.target_mode,
        device=device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
