#!/usr/bin/env python
"""Create a deterministic train/validation hold-out split for H0 datasets."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


def split_indices(labels: list[bool], val_ratio: float, seed: int, stratify: bool) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    if stratify:
        groups = {False: [], True: []}
        for idx, label in enumerate(labels):
            groups[bool(label)].append(idx)
        train_indices = []
        val_indices = []
        for indices in groups.values():
            rng.shuffle(indices)
            n_val = max(1, round(len(indices) * val_ratio)) if len(indices) > 1 else len(indices)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
    else:
        indices = list(range(len(labels)))
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_ratio))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
    return sorted(train_indices), sorted(val_indices)


def subset_payload(data: dict, indices: list[int], split_name: str, source: Path, val_ratio: float, seed: int) -> dict:
    result = dict(data)
    metadata = dict(data.get("metadata", {}))
    metadata.update(
        {
            "source": str(source),
            "holdout_split": split_name,
            "holdout_val_ratio": val_ratio,
            "holdout_seed": seed,
            "holdout_num_items": len(indices),
        }
    )
    result["metadata"] = metadata
    result["items"] = [data["items"][idx] for idx in indices]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    input_path = Path(args.input)
    data = torch.load(input_path, map_location="cpu")
    if "items" not in data or not data["items"]:
        raise ValueError(f"No H0 items found in {input_path}.")

    labels = [bool(item["is_hard"]) for item in data["items"]]
    train_indices, val_indices = split_indices(labels, args.val_ratio, args.seed, not args.no_stratify)
    if not train_indices or not val_indices:
        raise ValueError("Hold-out split produced an empty train or validation partition.")

    train_payload = subset_payload(data, train_indices, "train", input_path, args.val_ratio, args.seed)
    val_payload = subset_payload(data, val_indices, "valid", input_path, args.val_ratio, args.seed)

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_payload, train_output)
    torch.save(val_payload, val_output)

    def describe(name: str, payload: dict) -> str:
        labels = [bool(item["is_hard"]) for item in payload["items"]]
        hard_count = sum(labels)
        return f"{name}: total={len(labels)} hard={hard_count} hard_rate={hard_count / len(labels):.6f}"

    print(describe("train", train_payload))
    print(describe("valid", val_payload))


if __name__ == "__main__":
    sys.exit(main())
