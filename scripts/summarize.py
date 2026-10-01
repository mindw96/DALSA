#!/usr/bin/env python
"""Aggregate per-sample JSONL outputs into the paper's tables.

Every run directory under OUT (one level deep) that holds *.jsonl records becomes a row:
accuracy / average generated tokens per benchmark family (AIME24-26 = mean of the three AIME
sets), their equal-weight average, and the length-stop rate. Two extras:
  * if OUT/routing_all_short and OUT/routing_all_long exist, the Simple Routing threshold
    curve tau = 0.1..0.9 is composed offline from them (same generations, re-assigned);
  * DALSA runs also report the merge-bucket distribution and mean exact lambda (Table 4, Fig. 3).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

AIME = ("aime24", "aime25", "aime26")


def load_run(run_dir: Path) -> dict[tuple, dict]:
    records = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = [run_dir / Path(ds["output"]).name for ds in summary["datasets"]]
    else:
        paths = sorted(run_dir.glob("*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["dataset"], r.get("split"), r["id"], r.get("generation_index", 0))
                if key in records:
                    raise ValueError(f"Duplicate evaluation sample {key} in {path}")
                records[key] = r
    return records


def per_dataset(records) -> dict[str, dict]:
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)
    return {
        ds: {
            "acc": 100.0 * sum(bool(r["is_correct"]) for r in rs) / len(rs),
            "tokens": sum(float(r["num_tokens"]) for r in rs) / len(rs),
            "length_stop": 100.0 * sum(r.get("finish_reason") == "length" for r in rs) / len(rs),
            "n": len(rs),
        }
        for ds, rs in by_ds.items()
    }


def families(stats: dict[str, dict]) -> dict[str, dict]:
    if not stats:
        return {}
    fam = dict(stats)
    if all(a in stats for a in AIME):
        for name in AIME:
            del fam[name]
        fam["aime24_26"] = {k: sum(stats[a][k] for a in AIME) / 3 for k in ("acc", "tokens", "length_stop")}
    names = list(fam)
    fam["avg"] = {k: sum(fam[n][k] for n in names) / len(names) for k in ("acc", "tokens")}
    return fam


def routing_curve(short: dict, long: dict) -> dict[str, dict]:
    if short.keys() != long.keys():
        raise ValueError("routing_all_short and routing_all_long cover different samples")
    scores = {}
    for key, row in short.items():
        score = row.get("routing_score", row.get("p_hard"))
        long_score = long[key].get("routing_score", long[key].get("p_hard"))
        if score is None or long_score is None or not math.isclose(float(score), float(long_score), abs_tol=1e-6):
            raise ValueError(f"Missing or inconsistent routing score for {key}")
        scores[key] = float(score)
    curve = {}
    for tau in [round(0.1 * i, 1) for i in range(1, 10)]:
        chosen = [
            long[k] if scores[k] >= tau else s
            for k, s in short.items()
        ]
        curve[f"tau={tau}"] = families(per_dataset(chosen))
    return curve


def bucket_stats(records) -> dict[str, dict]:
    by_ds = defaultdict(list)
    for r in records:
        if "lambda_scale_exact" in r:
            by_ds[r["dataset"]].append(r)
    return {
        ds: {
            "buckets": dict(sorted(Counter(f"{float(r['lambda_scale']):.1f}" for r in rs).items())),
            "mean_exact_lambda": sum(float(r["lambda_scale_exact"]) for r in rs) / len(rs),
        }
        for ds, rs in by_ds.items()
    }


def summarize(out: Path) -> dict:
    runs = {d.name: load_run(d) for d in sorted(out.iterdir()) if d.is_dir() and any(d.glob("*.jsonl"))}
    versions = {r.get("scorer_version", "legacy-unversioned") for recs in runs.values() for r in recs.values()}
    if len(versions) > 1:
        raise ValueError("Cannot combine results from different scorer versions; use separate output roots")
    result = {}
    for name, recs in runs.items():
        if recs:
            stats = per_dataset(recs.values())
            result[name] = {"datasets": stats, "families": families(stats),
                            "scorer_version": next(iter(versions))}
    for name, recs in runs.items():
        buckets = bucket_stats(recs.values())
        if buckets:
            result[name]["buckets"] = buckets
    if "routing_all_short" in runs and "routing_all_long" in runs:
        result["routing_threshold_curve"] = routing_curve(runs["routing_all_short"], runs["routing_all_long"])
    return result


def print_table(result: dict) -> None:
    rows = {name: r["families"] for name, r in result.items() if "families" in r}
    rows.update(result.get("routing_threshold_curve", {}))
    cols = sorted({c for fam in rows.values() for c in fam if c != "avg"}) + ["avg"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, fam in rows.items():
        cells = [f"{fam[c]['acc']:.2f} ({fam[c]['tokens']:,.1f})" if c in fam else "-" for c in cols]
        print(f"| {name} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="Pipeline output root (contains one directory per run).")
    args = parser.parse_args()
    out = Path(args.out)
    result = summarize(out)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_table(result)


if __name__ == "__main__":
    main()
