#!/usr/bin/env python
"""Merge JSONL/summary outputs from sharded adaptive vLLM evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def shard_settings(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("method", "prompt_mode", "short_model", "long_model", "sampling", "vllm", "lambda", "scorer_version", "provenance")
    common = {key: summary.get(key) for key in keys}
    if common["sampling"] is not None:
        common["sampling"] = {key: value for key, value in common["sampling"].items() if key != "shard_index"}
    common["num_shards"] = summary.get("shard", {}).get("num_shards")
    common["routing_threshold"] = summary.get("plan", {}).get("routing_threshold")
    return common


def merge_shards(inputs: list[str], output: str) -> dict[str, Any]:
    summaries = [load_summary(path) for path in inputs]
    if not summaries:
        raise ValueError("No evaluation shards supplied.")

    output_dir = Path(output)
    merged_by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    elapsed_by_dataset: dict[tuple[str, str], list[float]] = defaultdict(list)

    reference_settings = shard_settings(summaries[0])
    seen = set()
    shard_indices = []
    for summary in summaries:
        if shard_settings(summary) != reference_settings:
            raise ValueError("Evaluation shards have different model, scorer, sampling, or merge settings.")
        if "shard" in summary:
            shard_indices.append(summary["shard"]["shard_index"])
        for dataset_summary in summary["datasets"]:
            key = (dataset_summary["dataset"], dataset_summary["split"])
            rows = read_jsonl(dataset_summary["output"])
            for row in rows:
                if (row["dataset"], row["split"]) != key:
                    raise ValueError("Evaluation row dataset/split differs from its summary.")
                if row.get("scorer_version") != summary.get("scorer_version"):
                    raise ValueError("Evaluation row scorer_version differs from its summary.")
                record_key = (*key, row["id"], row.get("generation_index", 0))
                if record_key in seen:
                    raise ValueError(f"Duplicate evaluation record: {record_key!r}")
                seen.add(record_key)
            merged_by_dataset[key].extend(rows)
            elapsed_by_dataset[key].append(float(dataset_summary.get("elapsed_seconds", 0.0)))

    if shard_indices:
        count = summaries[0]["shard"]["num_shards"]
        if not isinstance(count, int) or count < 1 or sorted(shard_indices) != list(range(count)):
            raise ValueError("Evaluation inputs must cover each shard_index in 0..num_shards-1 exactly once.")
    # Validate every input before creating or replacing any output.
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summaries = []
    for key, rows in merged_by_dataset.items():
        dataset, split = key
        rows.sort(key=lambda row: (int(row.get("index", 0)), int(row.get("generation_index", 0))))
        dataset_file = dataset.replace("/", "_")
        split_file = split.replace("/", "_")
        output_path = output_dir / f"{dataset_file}_{split_file}_{summaries[0]['method']}_{summaries[0]['prompt_mode']}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        finish_reasons: dict[str, int] = defaultdict(int)
        item_correct = {}
        lengths = []
        correct = 0
        for row in rows:
            ok = bool(row["is_correct"])
            correct += int(ok)
            lengths.append(int(row["num_tokens"]))
            finish_reasons[str(row.get("finish_reason"))] += 1
            item_key = row["id"]
            item_correct[item_key] = item_correct.get(item_key, False) or ok

        elapsed_parallel = max(elapsed_by_dataset[key]) if elapsed_by_dataset[key] else 0.0
        total_tokens = int(np.sum(lengths)) if lengths else 0
        dataset_summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "total": len(item_correct),
                "sample_count": len(rows),
                "correct_samples": correct,
                "any_correct": int(sum(int(v) for v in item_correct.values())),
                "accuracy": correct / len(rows) if rows else 0.0,
                "any_correct_rate": (sum(int(v) for v in item_correct.values()) / len(item_correct))
                if item_correct
                else 0.0,
                "avg_tokens": float(np.mean(lengths)) if lengths else 0.0,
                "total_tokens": total_tokens,
                "finish_reasons": dict(finish_reasons),
                "elapsed_seconds": elapsed_parallel,
                "tokens_per_second": float(total_tokens / elapsed_parallel) if elapsed_parallel > 0 else 0.0,
                "output": str(output_path),
            }
        )

    merged = dict(summaries[0])
    merged["datasets"] = dataset_summaries
    merged["merged_from"] = inputs
    merged.pop("shard", None)
    if "sampling" in merged:
        merged["sampling"] = {key: value for key, value in merged["sampling"].items() if key != "shard_index"}
    merged["timing"] = {
        **dict(merged.get("timing", {})),
        "parallel_generation_seconds_by_dataset": {
            f"{dataset}_{split}": max(values) for (dataset, split), values in elapsed_by_dataset.items()
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard summary.json paths or directories.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    merged = merge_shards(args.inputs, args.output_dir)
    print(json.dumps({"output": str(Path(args.output_dir) / "summary.json"), "datasets": merged["datasets"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
