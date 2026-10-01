#!/usr/bin/env python
"""Build a shared-prompt calibration pool (used for the CSQA/ARC-Challenge diagnostic)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dalsa.data import load_benchmark
from dalsa.prompts import render_shared_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="csqa,arc")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-dataset", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        items = load_benchmark(dataset, args.split, seed=args.seed)
        random.Random(args.seed + sum(ord(ch) for ch in dataset)).shuffle(items)
        for item in items[: args.per_dataset]:
            rows.append({"dataset": dataset, "split": args.split, "id": item["id"], "text": render_shared_prompt(item)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} calibration texts to {out}")


if __name__ == "__main__":
    main()
