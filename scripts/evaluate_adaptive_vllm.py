#!/usr/bin/env python
"""vLLM evaluator for routing, fixed merge, and bucketed DALSA.

The exact per-example DALSA path changes full model weights for every input,
which prevents vLLM from batching requests. This script first computes
P(Hard | x) in batches, then groups examples by routing target or lambda bucket.
Each group is evaluated with vLLM in one batched call.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from dalsa.answers import SCORER_VERSION, is_correct
from dalsa.data import load_benchmark
from dalsa.estimator import DifficultyEstimator
from dalsa.merge import (
    ACMCalibrationInfo,
    FIMCalibrationInfo,
    materialize_acm_merged_model,
    materialize_fim_merged_model,
    materialize_full_task_arithmetic_model,
    materialize_sparse_task_arithmetic_model,
    precompute_acm_calibration,
    precompute_fim_calibration,
)
from dalsa.modeling import load_causal_lm, load_tokenizer, set_seed
from dalsa.prompts import render_prompt


@dataclass
class EvalItem:
    global_index: int
    dataset: str
    split: str
    dataset_index: int
    item: dict[str, Any]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_years(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_lambda_values(value: str | None, min_lambda: float, max_lambda: float, buckets: int) -> list[float]:
    if value:
        lambdas = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        if buckets < 2:
            lambdas = [float(min_lambda)]
        else:
            lambdas = [float(x) for x in np.linspace(min_lambda, max_lambda, buckets)]
    if not lambdas:
        raise ValueError("At least one lambda bucket is required.")
    return sorted(set(round(v, 6) for v in lambdas))


def lambda_from_hard_prob(p_hard: float, min_lambda: float, max_lambda: float, merge_base: str) -> float:
    if merge_base == "long":
        return float(max_lambda - (max_lambda - min_lambda) * p_hard)
    return float(min_lambda + (max_lambda - min_lambda) * p_hard)


def identity_lambda_score_remap() -> dict[str, Any]:
    return {"mode": "none"}


def apply_lambda_score_remap(score: float, remap_info: dict[str, Any]) -> float:
    if remap_info.get("mode", "none") == "none":
        return float(score)
    if remap_info["mode"] == "quantile":
        lo = float(remap_info["score_lo"])
        hi = float(remap_info["score_hi"])
        remapped = (float(score) - lo) / max(hi - lo, 1e-12)
        return float(np.clip(remapped, 0.0, 1.0))
    raise ValueError(f"Unsupported lambda score remap mode: {remap_info.get('mode')}")


@torch.no_grad()
def build_lambda_score_remap(args) -> dict[str, Any]:
    if args.lambda_score_remap == "none":
        return identity_lambda_score_remap()
    if args.method not in {"routing", "dalsa_bucketed"}:
        raise ValueError("--lambda-score-remap is only supported for method=routing or dalsa_bucketed.")
    if not args.lambda_remap_reference_h0:
        raise ValueError("--lambda-remap-reference-h0 is required for --lambda-score-remap quantile.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DifficultyEstimator.load(args.estimator, device=device)
    h0_data = torch.load(args.lambda_remap_reference_h0, map_location="cpu")
    if "items" not in h0_data or not h0_data["items"]:
        raise ValueError(f"No H0 items found in {args.lambda_remap_reference_h0}.")

    scores = []
    batch_size = int(args.lambda_remap_batch_size)
    for start in tqdm(range(0, len(h0_data["items"]), batch_size), desc="fit lambda score remap"):
        batch = h0_data["items"][start : start + batch_size]
        h0 = torch.stack([item["h0"].float() for item in batch])
        scores.extend(float(x) for x in estimator.predict_batch_proba_hard(h0))
    arr = np.asarray(scores, dtype=np.float64)
    lo = float(np.quantile(arr, args.lambda_remap_q_low))
    hi = float(np.quantile(arr, args.lambda_remap_q_high))
    if hi <= lo:
        hi = lo + 1e-6

    del estimator
    torch.cuda.empty_cache()
    return {
        "mode": "quantile",
        "reference_h0": args.lambda_remap_reference_h0,
        "q_low": float(args.lambda_remap_q_low),
        "q_high": float(args.lambda_remap_q_high),
        "score_lo": lo,
        "score_hi": hi,
        "reference_count": int(arr.size),
        "reference_mean": float(np.mean(arr)),
        "reference_std": float(np.std(arr)),
    }


def lambda_semantics(args) -> str:
    if args.merge_scope in {"ties_full", "dare_full"}:
        mode = "TIES-trimmed" if args.merge_scope == "ties_full" else "DARE-dropped"
        if args.merge_base == "long":
            return f"{mode} full-model TA short-minus-long delta added to long base"
        return f"{mode} full-model TA long delta added to short base"
    if args.merge_scope == "fim_full" and args.merge_base == "long":
        return "FIM-weighted full-model TA short-minus-long delta added to long base"
    if args.merge_scope == "fim_full":
        return "FIM-weighted full-model TA long delta added to short base"
    if args.merge_scope == "full_model" and args.merge_base == "long":
        return "full-model raw task arithmetic short-minus-long delta added to long base"
    if args.merge_scope == "full_model":
        return "full-model raw task arithmetic long delta added to short base"
    if args.disable_layer_consensus and args.merge_base == "long":
        return "raw task arithmetic short-minus-long delta added to long base"
    if args.disable_layer_consensus:
        return "ACM-target raw task arithmetic long delta added to short base"
    if args.merge_base == "long":
        return "amount of short-minus-long delta added to long base"
    return "amount of long delta added to short base"


def nearest_lambda(value: float, lambda_values: list[float]) -> float:
    return min(lambda_values, key=lambda x: abs(x - value))


def safe_slug(value: str, max_len: int = 54) -> str:
    keep = []
    for ch in value:
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in {".", "-", "_"}:
            keep.append(ch)
        else:
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:max_len] or "model"


def load_eval_items(args) -> list[EvalItem]:
    datasets = parse_csv(args.datasets)
    splits = parse_csv(args.splits) if args.splits else ["test"] * len(datasets)
    if len(splits) != len(datasets):
        raise ValueError("--splits must have the same length as --datasets.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    rows: list[EvalItem] = []
    for dataset, split in zip(datasets, splits):
        items = load_benchmark(
            dataset,
            split,
            limit=args.limit,
            seed=args.seed,
            aime_years=parse_years(args.aime_years),
        )
        for idx, item in enumerate(items):
            if idx % args.num_shards != args.shard_index:
                continue
            rows.append(
                EvalItem(
                    global_index=len(rows),
                    dataset=dataset,
                    split=split,
                    dataset_index=idx,
                    item=item,
                )
            )
    return rows


@torch.no_grad()
def extract_h0_batch(model, tokenizer, prompts: list[str], max_length: int) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]
    if getattr(tokenizer, "padding_side", "right") == "left":
        last_indices = torch.full(
            (len(prompts),),
            inputs["attention_mask"].shape[1] - 1,
            device=hidden.device,
            dtype=torch.long,
        )
    else:
        last_indices = inputs["attention_mask"].sum(dim=1) - 1
    return torch.stack([hidden[i, last_indices[i]].detach().cpu().float() for i in range(len(prompts))])


def compute_p_hard(args, rows: list[EvalItem]) -> tuple[dict[int, float], float, float]:
    if not args.estimator:
        raise ValueError(f"--estimator is required for method={args.method}")

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DifficultyEstimator.load(args.estimator, device=device)
    threshold = float(estimator.threshold)
    tokenizer = load_tokenizer(args.short_model)
    model = load_causal_lm(args.short_model)
    model.eval()

    p_hard_by_index: dict[int, float] = {}
    for start in tqdm(range(0, len(rows), args.h0_batch_size), desc="batch P(Hard)"):
        batch = rows[start : start + args.h0_batch_size]
        prompts = [render_prompt(tokenizer, args.short_model, row.item, "shared") for row in batch]
        h0 = extract_h0_batch(model, tokenizer, prompts, args.h0_max_length)
        probs = estimator.predict_batch_proba_hard(h0)
        for row, prob in zip(batch, probs):
            p_hard_by_index[row.global_index] = float(prob)

    del model
    del estimator
    gc.collect()
    torch.cuda.empty_cache()
    return p_hard_by_index, time.perf_counter() - started, threshold


def calibration_request(args) -> dict[str, Any]:
    request = {"scope": args.merge_scope, "short_model": args.short_model,
               "long_model": args.long_model, "seed": args.seed}
    if args.merge_scope == "fim_full":
        request.update(fim_settings(args))
    else:
        request.update(theta=args.theta, dataset=args.calibration_dataset,
                       split=args.calibration_split, samples=args.calibration_samples,
                       mi_clusters=args.mi_clusters, max_length=1024)
        source = Path(args.calibration_dataset)
        if source.is_file():
            with source.open("rb") as stream:
                request["dataset_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return request


def validate_calibration(info, args) -> None:
    if info.metadata.get("request") != calibration_request(args):
        raise ValueError("Calibration cache settings differ or lack request provenance; "
                         "use a new --calibration path (or a new OUT directory).")


def get_calibration(args) -> ACMCalibrationInfo:
    if args.calibration and Path(args.calibration).exists():
        info = ACMCalibrationInfo.load(args.calibration)
        validate_calibration(info, args)
        return info
    info = precompute_acm_calibration(
        long_model_name=args.long_model,
        short_model_name=args.short_model,
        theta=args.theta,
        dataset_name=args.calibration_dataset,
        split=args.calibration_split,
        max_samples=args.calibration_samples,
        mi_clusters=args.mi_clusters,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    info.metadata["request"] = calibration_request(args)
    if args.calibration:
        info.save(args.calibration)
    return info


def get_fim_calibration(args) -> FIMCalibrationInfo:
    if args.calibration and Path(args.calibration).exists():
        info = FIMCalibrationInfo.load(args.calibration)
        validate_calibration(info, args)
        return info
    info = precompute_fim_calibration(
        long_model_name=args.long_model,
        short_model_name=args.short_model,
        random_samples=args.fim_random_samples,
        random_seq_len=args.fim_seq_len,
        random_batch_size=args.fim_batch_size,
        alpha_min=args.fim_alpha_min,
        alpha_max=args.fim_alpha_max,
        temperature=args.fim_temperature,
        seed=args.seed,
        device=args.fim_device if torch.cuda.is_available() else "cpu",
    )
    info.metadata["request"] = calibration_request(args)
    if args.calibration:
        info.save(args.calibration)
    return info


def fim_settings(args) -> dict[str, Any] | None:
    if args.merge_scope != "fim_full":
        return None
    return {
        "random_samples": args.fim_random_samples,
        "seq_len": args.fim_seq_len,
        "batch_size": args.fim_batch_size,
        "alpha_min": args.fim_alpha_min,
        "alpha_max": args.fim_alpha_max,
        "temperature": args.fim_temperature,
        "device": args.fim_device,
    }


def merged_cache_path(args, lambda_scale: float) -> Path:
    payload_data = {
        "cache_version": 2,
        "short": args.short_model,
        "long": args.long_model,
        "calibration": args.calibration,
        "lambda": round(float(lambda_scale), 6),
        "disable_layer_consensus": bool(args.disable_layer_consensus),
    }
    if args.merge_scope != "acm_targets":
        payload_data["merge_scope"] = args.merge_scope
    if args.merge_scope in {"acm_targets", "fim_full"}:
        payload_data["calibration_request"] = calibration_request(args)
        if args.calibration and Path(args.calibration).is_file():
            path = Path(args.calibration).resolve()
            stat = path.stat()
            # ponytail: local file version, avoiding a multi-GB hash per lambda.
            payload_data["calibration_file"] = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if args.merge_scope == "ties_full":
        payload_data["ties_density"] = round(float(args.ties_density), 6)
    if args.merge_scope == "dare_full":
        payload_data["dare_drop_rate"] = round(float(args.dare_drop_rate), 6)
        payload_data["dare_seed"] = int(args.seed)
    if args.merge_base != "short":
        payload_data["merge_base"] = args.merge_base
    payload = json.dumps(payload_data, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    prefix = f"{safe_slug(args.short_model)}__{safe_slug(args.long_model)}"
    if args.merge_scope == "full_model":
        prefix += "__full-ta"
    elif args.merge_scope == "fim_full":
        prefix += "__fim-ta"
    elif args.merge_scope == "ties_full":
        prefix += "__ties-ta"
    elif args.merge_scope == "dare_full":
        prefix += "__dare-ta"
    if args.merge_base != "short":
        prefix += f"__base-{args.merge_base}"
    if args.disable_layer_consensus:
        prefix += "__raw-ta"
    name = f"{prefix}__lambda-{lambda_scale:.4f}__{digest}"
    return Path(args.merged_model_cache) / name


def prepare_merged_models(args, lambda_values: list[float]) -> tuple[dict[float, Path], float]:
    started = time.perf_counter()
    paths: dict[float, Path] = {}
    if args.merge_scope == "full_model":
        for lambda_scale in lambda_values:
            target = merged_cache_path(args, lambda_scale)
            paths[lambda_scale] = materialize_full_task_arithmetic_model(
                short_model_name=args.short_model,
                long_model_name=args.long_model,
                lambda_scale=lambda_scale,
                output_dir=target,
                force=args.force_rematerialize,
                max_shard_size=args.max_shard_size,
                merge_base=args.merge_base,
            )
    elif args.merge_scope in {"ties_full", "dare_full"}:
        for lambda_scale in lambda_values:
            target = merged_cache_path(args, lambda_scale)
            paths[lambda_scale] = materialize_sparse_task_arithmetic_model(
                short_model_name=args.short_model,
                long_model_name=args.long_model,
                lambda_scale=lambda_scale,
                output_dir=target,
                mode="ties" if args.merge_scope == "ties_full" else "dare",
                ties_density=args.ties_density,
                dare_drop_rate=args.dare_drop_rate,
                seed=args.seed,
                force=args.force_rematerialize,
                max_shard_size=args.max_shard_size,
                merge_base=args.merge_base,
            )
    elif args.merge_scope == "fim_full":
        fim_info = get_fim_calibration(args)
        for lambda_scale in lambda_values:
            target = merged_cache_path(args, lambda_scale)
            paths[lambda_scale] = materialize_fim_merged_model(
                fim_info,
                lambda_scale=lambda_scale,
                output_dir=target,
                force=args.force_rematerialize,
                max_shard_size=args.max_shard_size,
                merge_base=args.merge_base,
            )
        del fim_info
    else:
        calib = get_calibration(args)
        for lambda_scale in lambda_values:
            target = merged_cache_path(args, lambda_scale)
            paths[lambda_scale] = materialize_acm_merged_model(
                calib,
                lambda_scale=lambda_scale,
                output_dir=target,
                force=args.force_rematerialize,
                max_shard_size=args.max_shard_size,
                merge_base=args.merge_base,
                disable_layer_consensus=args.disable_layer_consensus,
            )
        del calib
    gc.collect()
    torch.cuda.empty_cache()
    return paths, time.perf_counter() - started


def shutdown_vllm(llm) -> None:
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def make_sampling_params(args):
    from vllm import SamplingParams

    return SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def load_vllm(args, model_path: str):
    from vllm import LLM

    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    return LLM(**kwargs)


def init_output_files(args, rows: list[EvalItem]) -> dict[tuple[str, str], Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    shard_suffix = ""
    if args.num_shards > 1:
        shard_suffix = f"_shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    for row in rows:
        key = (row.dataset, row.split)
        if key not in outputs:
            path = output_dir / f"{row.dataset.replace('/', '_')}_{row.split.replace('/', '_')}_{args.method}_{args.prompt_mode}{shard_suffix}.jsonl"
            if path.exists():
                path.unlink()
            outputs[key] = path
    return outputs


def empty_metrics(rows: list[EvalItem]) -> dict[tuple[str, str], dict[str, Any]]:
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.dataset, row.split)
        if key not in metrics:
            metrics[key] = {
                "dataset": row.dataset,
                "split": row.split,
                "total": 0,
                "sample_count": 0,
                "correct_samples": 0,
                "any_correct": 0,
                "lengths": [],
                "finish_reasons": defaultdict(int),
                "item_correct": {},
                "elapsed_seconds": 0.0,
            }
        metrics[key]["total"] += 1
    return metrics


def update_metrics(metric: dict[str, Any], row: EvalItem, sample_ok: bool, n_tokens: int, finish_reason: Any) -> None:
    metric["sample_count"] += 1
    metric["correct_samples"] += int(sample_ok)
    metric["lengths"].append(n_tokens)
    metric["finish_reasons"][str(finish_reason)] += 1
    metric["item_correct"][row.global_index] = metric["item_correct"].get(row.global_index, False) or bool(sample_ok)


def finalize_metrics(metrics: dict[tuple[str, str], dict[str, Any]], output_paths: dict[tuple[str, str], Path]) -> list[dict[str, Any]]:
    summaries = []
    for key, metric in metrics.items():
        lengths = metric.pop("lengths")
        item_correct = metric.pop("item_correct")
        finish_reasons = dict(metric.pop("finish_reasons"))
        metric["any_correct"] = int(sum(int(v) for v in item_correct.values()))
        metric["accuracy"] = metric["correct_samples"] / metric["sample_count"] if metric["sample_count"] else 0.0
        metric["any_correct_rate"] = metric["any_correct"] / metric["total"] if metric["total"] else 0.0
        metric["avg_tokens"] = float(np.mean(lengths)) if lengths else 0.0
        metric["total_tokens"] = int(np.sum(lengths)) if lengths else 0
        metric["tokens_per_second"] = (
            float(metric["total_tokens"] / metric["elapsed_seconds"]) if metric["elapsed_seconds"] > 0 else 0.0
        )
        metric["finish_reasons"] = finish_reasons
        metric["output"] = str(output_paths[key])
        summaries.append(metric)
    return summaries


def evaluate_group(
    args,
    *,
    group_name: str,
    model_path: str,
    prompt_model_name: str,
    rows: list[EvalItem],
    output_paths: dict[tuple[str, str], Path],
    metrics: dict[tuple[str, str], dict[str, Any]],
    extra_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if not rows:
        return {
            "group": group_name,
            "model": model_path,
            "num_items": 0,
            "load_seconds": 0.0,
            "generation_seconds": 0.0,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [render_prompt(tokenizer, prompt_model_name, row.item, args.prompt_mode) for row in rows]

    load_started = time.perf_counter()
    llm = load_vllm(args, model_path)
    load_seconds = time.perf_counter() - load_started

    sampling = make_sampling_params(args)
    gen_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    generation_seconds = time.perf_counter() - gen_started

    for row, prompt, output in tqdm(list(zip(rows, prompts, outputs)), desc=f"score {group_name}"):
        item_correct = False
        for gen_idx, sample in enumerate(output.outputs):
            response = sample.text
            ok, pred = is_correct(response, row.item)
            item_correct = item_correct or ok
            n_tokens = len(sample.token_ids)
            finish_reason = getattr(sample, "finish_reason", None)
            key = (row.dataset, row.split)
            update_metrics(metrics[key], row, ok, n_tokens, finish_reason)
            record = {
                "index": row.dataset_index,
                "global_index": row.global_index,
                "generation_index": gen_idx,
                "id": row.item["id"],
                "dataset": row.dataset,
                "split": row.split,
                "task_type": row.item.get("task_type"),
                "question": row.item["question"],
                "ground_truth": row.item["ground_truth"],
                "predicted_answer": pred,
                "is_correct": ok,
                "scorer_version": SCORER_VERSION,
                "num_tokens": n_tokens,
                "finish_reason": finish_reason,
                "method": args.method,
                "group": group_name,
                "model": model_path,
                "prompt_model_name": prompt_model_name,
                "prompt_mode": args.prompt_mode,
                "prompt": prompt,
                "response": response,
                **extra_by_index.get(row.global_index, {}),
            }
            with open(output_paths[key], "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics[(row.dataset, row.split)]["item_correct"][row.global_index] = item_correct
    for row in rows:
        metrics[(row.dataset, row.split)]["elapsed_seconds"] += generation_seconds * (1.0 / len(rows))

    shutdown_vllm(llm)
    return {
        "group": group_name,
        "model": model_path,
        "num_items": len(rows),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }


def assign_groups(
    args,
    rows: list[EvalItem],
    p_hard_by_index: dict[int, float],
    routing_threshold: float,
    lambda_score_remap: dict[str, Any] | None = None,
) -> tuple[dict[str, list[EvalItem]], dict[int, dict[str, Any]], list[float]]:
    groups: dict[str, list[EvalItem]] = defaultdict(list)
    extra_by_index: dict[int, dict[str, Any]] = {}
    lambda_values: list[float] = []
    if lambda_score_remap is None:
        lambda_score_remap = identity_lambda_score_remap()

    if args.method == "routing":
        for row in rows:
            p_hard = p_hard_by_index[row.global_index]
            routing_score = apply_lambda_score_remap(p_hard, lambda_score_remap)
            routed_to = "long" if routing_score >= routing_threshold else "short"
            groups[routed_to].append(row)
            extra_by_index[row.global_index] = {
                "p_hard": p_hard,
                "routing_score": routing_score,
                "routing_score_remap": lambda_score_remap.get("mode", "none"),
                "routing_threshold": routing_threshold,
                "routed_to": routed_to,
            }
    elif args.method == "fixed_merge":
        lambda_values = [round(float(args.fixed_lambda), 6)]
        for row in rows:
            groups[f"lambda_{args.fixed_lambda:.4f}"].append(row)
            extra_by_index[row.global_index] = {"lambda_scale": float(args.fixed_lambda)}
    elif args.method == "dalsa_bucketed":
        lambda_values = parse_lambda_values(args.lambda_values, args.min_lambda, args.max_lambda, args.lambda_buckets)
        for row in rows:
            p_hard = p_hard_by_index[row.global_index]
            lambda_score = apply_lambda_score_remap(p_hard, lambda_score_remap)
            exact_lambda = lambda_from_hard_prob(lambda_score, args.min_lambda, args.max_lambda, args.merge_base)
            bucket_lambda = nearest_lambda(exact_lambda, lambda_values)
            group_name = f"lambda_{bucket_lambda:.4f}"
            groups[group_name].append(row)
            extra_by_index[row.global_index] = {
                "p_hard": p_hard,
                "lambda_score": lambda_score,
                "lambda_score_remap": lambda_score_remap.get("mode", "none"),
                "lambda_scale_exact": exact_lambda,
                "lambda_scale": bucket_lambda,
                "lambda_bucket_error": abs(bucket_lambda - exact_lambda),
                "lambda_values": lambda_values,
            }
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    return dict(groups), extra_by_index, lambda_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["routing", "fixed_merge", "dalsa_bucketed"], required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--short-model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--long-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--estimator", default=None)
    parser.add_argument("--prompt-mode", choices=["shared", "native"], default="shared")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aime-years", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--h0-batch-size", type=int, default=16)
    parser.add_argument("--h0-max-length", type=int, default=2048)
    parser.add_argument("--routing-threshold", type=float, default=None)
    parser.add_argument("--min-lambda", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.5)
    parser.add_argument("--fixed-lambda", type=float, default=0.7)
    parser.add_argument("--lambda-buckets", type=int, default=5)
    parser.add_argument("--lambda-values", default=None, help="Optional comma-separated bucket centers.")
    parser.add_argument(
        "--lambda-score-remap",
        choices=["none", "quantile"],
        default="none",
        help="Optional remap applied to estimator scores before converting them to lambda buckets.",
    )
    parser.add_argument(
        "--lambda-remap-reference-h0",
        default=None,
        help="H0 .pt file used to fit score quantiles for --lambda-score-remap quantile.",
    )
    parser.add_argument("--lambda-remap-q-low", type=float, default=0.05)
    parser.add_argument("--lambda-remap-q-high", type=float, default=0.95)
    parser.add_argument("--lambda-remap-batch-size", type=int, default=256)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--calibration-dataset", default="simplescaling/s1K")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-samples", type=int, default=20)
    parser.add_argument("--mi-clusters", type=int, default=8)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--merged-model-cache", default="checkpoints/merged_vllm")
    parser.add_argument("--merge-base", choices=["short", "long"], default="short")
    parser.add_argument(
        "--merge-scope",
        choices=["acm_targets", "full_model", "fim_full", "ties_full", "dare_full"],
        default="acm_targets",
        help=(
            "Merge only ACM target modules, raw full-model TA, FIM-weighted full-model TA, "
            "or sparsified full-model TA (TIES magnitude trimming / DARE random drop)."
        ),
    )
    parser.add_argument("--ties-density", type=float, default=0.2)
    parser.add_argument("--dare-drop-rate", type=float, default=0.5)
    parser.add_argument(
        "--disable-layer-consensus",
        action="store_true",
        help="Use raw task arithmetic without ACM layer-wise consensus scaling.",
    )
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--fim-random-samples", type=int, default=8)
    parser.add_argument("--fim-seq-len", type=int, default=128)
    parser.add_argument("--fim-batch-size", type=int, default=1)
    parser.add_argument("--fim-alpha-min", type=float, default=0.0)
    parser.add_argument("--fim-alpha-max", type=float, default=1.0)
    parser.add_argument("--fim-temperature", type=float, default=1.0)
    parser.add_argument("--fim-device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    if args.method in {"fixed_merge", "dalsa_bucketed"} and args.prompt_mode != "shared":
        raise ValueError("Merge methods must use --prompt-mode shared to avoid chat-token mismatch.")
    if args.method in {"routing", "dalsa_bucketed"} and not args.estimator:
        raise ValueError(f"--estimator is required for method={args.method}")
    if args.materialize_only and args.method not in {"fixed_merge", "dalsa_bucketed"}:
        raise ValueError("--materialize-only is only supported for merge methods.")
    if not 0.0 <= args.lambda_remap_q_low < args.lambda_remap_q_high <= 1.0:
        raise ValueError("--lambda-remap-q-low/high must satisfy 0 <= low < high <= 1.")

    started = time.perf_counter()
    if args.materialize_only:
        if args.method == "fixed_merge":
            lambda_values = [round(float(args.fixed_lambda), 6)]
        else:
            lambda_values = parse_lambda_values(args.lambda_values, args.min_lambda, args.max_lambda, args.lambda_buckets)
        merged_paths, materialize_seconds = prepare_merged_models(args, lambda_values)
        summary = {
            "scorer_version": SCORER_VERSION,
            "method": args.method,
            "short_model": args.short_model,
            "long_model": args.long_model,
            "prompt_mode": args.prompt_mode,
            "merge_base": args.merge_base,
            "disable_layer_consensus": args.disable_layer_consensus,
            "materialize_only": True,
            "lambda": {
                "min_lambda": args.min_lambda,
                "max_lambda": args.max_lambda,
                "fixed_lambda": args.fixed_lambda,
                "lambda_values": lambda_values,
                "bucketed": args.method == "dalsa_bucketed",
                "merge_base": args.merge_base,
                "merge_scope": args.merge_scope,
                "disable_layer_consensus": args.disable_layer_consensus,
                "lambda_semantics": lambda_semantics(args),
                "fim": fim_settings(args),
            },
            "materialized_models": {f"{value:.4f}": str(path) for value, path in merged_paths.items()},
            "timing": {
                "elapsed_seconds": time.perf_counter() - started,
                "materialize_seconds": materialize_seconds,
            },
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    rows = load_eval_items(args)
    p_hard_by_index: dict[int, float] = {}
    p_hard_seconds = 0.0
    estimator_threshold = None
    lambda_score_remap = identity_lambda_score_remap()
    if args.method in {"routing", "dalsa_bucketed"}:
        p_hard_by_index, p_hard_seconds, estimator_threshold = compute_p_hard(args, rows)
    if args.method in {"routing", "dalsa_bucketed"} and args.lambda_score_remap != "none":
        lambda_score_remap = build_lambda_score_remap(args)
    routing_threshold = (
        float(args.routing_threshold)
        if args.routing_threshold is not None
        else float(estimator_threshold if estimator_threshold is not None else 0.5)
    )

    groups, extra_by_index, lambda_values = assign_groups(
        args,
        rows,
        p_hard_by_index,
        routing_threshold,
        lambda_score_remap=lambda_score_remap,
    )
    plan = {
        "method": args.method,
        "total_items": len(rows),
        "groups": {name: len(items) for name, items in groups.items()},
        "lambda_values": lambda_values,
        "routing_threshold": routing_threshold if args.method == "routing" else None,
        "lambda_score_remap": lambda_score_remap,
        "p_hard_seconds": p_hard_seconds,
        "prompt_mode": args.prompt_mode,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    merged_paths: dict[float, Path] = {}
    materialize_seconds = 0.0
    if args.method in {"fixed_merge", "dalsa_bucketed"}:
        used_lambdas = [value for value in lambda_values if groups.get(f"lambda_{value:.4f}")]
        merged_paths, materialize_seconds = prepare_merged_models(args, used_lambdas)
        if args.prepare_only:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "scorer_version": SCORER_VERSION,
                "method": args.method,
                "short_model": args.short_model,
                "long_model": args.long_model,
                "prompt_mode": args.prompt_mode,
                "merge_base": args.merge_base,
                "disable_layer_consensus": args.disable_layer_consensus,
                "prepared_lambdas": used_lambdas,
                "merged_paths": {str(k): str(v) for k, v in merged_paths.items()},
                "merge_scope": args.merge_scope,
                "timing": {
                    "elapsed_seconds": time.perf_counter() - started,
                    "p_hard_seconds": p_hard_seconds,
                    "materialize_seconds": materialize_seconds,
                },
                "shard": {
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                },
                "plan": plan,
            }
            summary_path = output_dir / "prepare_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return

    output_paths = init_output_files(args, rows)
    metrics = empty_metrics(rows)
    group_summaries = []
    if args.method == "routing":
        if groups.get("short"):
            group_summaries.append(
                evaluate_group(
                    args,
                    group_name="short",
                    model_path=args.short_model,
                    prompt_model_name=args.short_model,
                    rows=groups["short"],
                    output_paths=output_paths,
                    metrics=metrics,
                    extra_by_index=extra_by_index,
                )
            )
        if groups.get("long"):
            group_summaries.append(
                evaluate_group(
                    args,
                    group_name="long",
                    model_path=args.long_model,
                    prompt_model_name=args.long_model,
                    rows=groups["long"],
                    output_paths=output_paths,
                    metrics=metrics,
                    extra_by_index=extra_by_index,
                )
            )
    else:
        for lambda_scale in lambda_values:
            group_name = f"lambda_{lambda_scale:.4f}"
            if not groups.get(group_name):
                continue
            group_summaries.append(
                evaluate_group(
                    args,
                    group_name=group_name,
                    model_path=str(merged_paths[lambda_scale]),
                    prompt_model_name=args.short_model,
                    rows=groups[group_name],
                    output_paths=output_paths,
                    metrics=metrics,
                    extra_by_index=extra_by_index,
                )
            )

    dataset_summaries = finalize_metrics(metrics, output_paths)
    summary = {
        "scorer_version": SCORER_VERSION,
        "method": args.method,
        "short_model": args.short_model,
        "long_model": args.long_model,
        "prompt_mode": args.prompt_mode,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_generations": args.num_generations,
            "seed": args.seed,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "vllm": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
        "lambda": {
            "min_lambda": args.min_lambda,
            "max_lambda": args.max_lambda,
            "fixed_lambda": args.fixed_lambda,
            "lambda_values": lambda_values,
            "bucketed": args.method == "dalsa_bucketed",
            "score_remap": lambda_score_remap,
            "merge_base": args.merge_base,
            "merge_scope": args.merge_scope,
            "disable_layer_consensus": args.disable_layer_consensus,
            "lambda_semantics": lambda_semantics(args),
            "fim": fim_settings(args),
        },
        "timing": {
            "elapsed_seconds": time.perf_counter() - started,
            "p_hard_seconds": p_hard_seconds,
            "materialize_seconds": materialize_seconds,
        },
        "groups": group_summaries,
        "datasets": dataset_summaries,
        "plan": plan,
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    }
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
