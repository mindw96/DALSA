#!/usr/bin/env python
"""Merge h0 .pt shards produced by extract_h0.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# These fields describe individual files or shard results, not their common settings.
SHARD_FIELDS = {"source", "merged_from", "shard_index", "total", "accuracy", "hard_count"}


def settings(metadata: dict) -> dict:
    return {
        key: settings(value) if isinstance(value, dict) else value
        for key, value in metadata.items() if key not in SHARD_FIELDS
    }


def merge_shards(inputs: list[str], output: str) -> dict:
    payloads = [torch.load(path, map_location="cpu") for path in inputs]
    if not payloads:
        raise ValueError("No h0 shards supplied.")
    first = payloads[0]
    metadata = first.get("metadata", {})
    if "num_rollouts" in metadata:
        k = metadata["num_rollouts"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("H0 metadata.num_rollouts must be a positive integer.")
    items = []
    seen = set()
    shard_indices = []
    for payload in payloads:
        if int(payload["hidden_dim"]) != int(first["hidden_dim"]):
            raise ValueError("All h0 shards must use the same hidden_dim.")
        shard_metadata = payload.get("metadata", {})
        if settings(shard_metadata) != settings(metadata):
            raise ValueError("All h0 shards must use the same metadata settings (including rollout K and scorer).")
        if "shard_index" in shard_metadata:
            shard_indices.append(shard_metadata["shard_index"])
        for item in payload["items"]:
            key = item["id"]
            if key in seen:
                raise ValueError(f"Duplicate h0 item id: {key!r}")
            seen.add(key)
            items.append(item)
    if shard_indices:
        count = metadata.get("num_shards")
        if not isinstance(count, int) or count < 1 or sorted(shard_indices) != list(range(count)):
            raise ValueError("H0 inputs must cover each shard_index in 0..num_shards-1 exactly once.")
    items.sort(key=lambda item: item["id"])
    result = {
        "metadata": {
            **{key: value for key, value in metadata.items() if key not in SHARD_FIELDS},
            "source": inputs,
            "merged_from": inputs,
        },
        "hidden_dim": int(first["hidden_dim"]),
        "items": items,
    }
    # Validate every input before creating or replacing any output.
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge_shards(args.inputs, args.output)
    print(f"saved {len(result['items'])} merged h0 vectors to {args.output}")


if __name__ == "__main__":
    main()
