#!/usr/bin/env python
"""Evaluate one model on one or more DALSA benchmarks with a single vLLM load."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

from dalsa.answers import SCORER_VERSION, is_correct
from dalsa.data import load_benchmark
from dalsa.modeling import set_seed
from dalsa.prompts import render_prompt


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_years(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_dataset(args, llm, tokenizer, dataset: str, split: str, output_dir: Path) -> dict:
    from vllm import SamplingParams

    items = load_benchmark(
        dataset,
        split,
        limit=args.limit,
        seed=args.seed,
        aime_years=parse_years(args.aime_years),
    )
    prompts = [render_prompt(tokenizer, args.model, item, args.prompt_mode) for item in items]
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - started

    output_path = output_dir / f"{dataset.replace('/', '_')}_{split.replace('/', '_')}_{args.prompt_mode}.jsonl"
    correct_samples = 0
    any_correct = 0
    lengths = []
    finish_reasons = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (item, prompt, output) in enumerate(tqdm(list(zip(items, prompts, outputs)), desc=f"score {dataset}")):
            item_correct = False
            for gen_idx, sample in enumerate(output.outputs):
                response = sample.text
                ok, pred = is_correct(response, item)
                n_tokens = len(sample.token_ids)
                finish_reason = getattr(sample, "finish_reason", None)
                finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
                correct_samples += int(ok)
                item_correct = item_correct or ok
                lengths.append(n_tokens)
                f.write(
                    json.dumps(
                        {
                            "index": idx,
                            "generation_index": gen_idx,
                            "id": item["id"],
                            "dataset": dataset,
                            "split": split,
                            "task_type": item.get("task_type"),
                            "question": item["question"],
                            "ground_truth": item["ground_truth"],
                            "predicted_answer": pred,
                            "is_correct": ok,
                            "scorer_version": SCORER_VERSION,
                            "num_tokens": n_tokens,
                            "finish_reason": finish_reason,
                            "prompt_mode": args.prompt_mode,
                            "prompt": prompt,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            any_correct += int(item_correct)

    sample_count = len(lengths)
    return {
        "dataset": dataset,
        "split": split,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "total": len(items),
        "num_generations": args.num_generations,
        "sample_count": sample_count,
        "correct_samples": correct_samples,
        "accuracy": correct_samples / sample_count if sample_count else 0.0,
        "any_correct": any_correct,
        "any_correct_rate": any_correct / len(items) if items else 0.0,
        "avg_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "total_tokens": int(np.sum(lengths)) if lengths else 0,
        "finish_reasons": finish_reasons,
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(np.sum(lengths) / elapsed) if elapsed > 0 else 0.0,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--splits", default=None, help="Comma-separated splits; defaults to test for every dataset.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-mode", choices=["shared", "native"], default="native")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aime-years", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    set_seed(args.seed)
    datasets = parse_csv(args.datasets)
    splits = parse_csv(args.splits) if args.splits else ["test"] * len(datasets)
    if len(splits) != len(datasets):
        raise ValueError("--splits must have the same length as --datasets.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_started = time.perf_counter()
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_started

    summaries = []
    for dataset, split in zip(datasets, splits):
        summaries.append(evaluate_dataset(args, llm, tokenizer, dataset, split, output_dir))

    summary = {
        "scorer_version": SCORER_VERSION,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "max_model_len": args.max_model_len,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_generations": args.num_generations,
            "seed": args.seed,
        },
        "vllm": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
        "load_seconds": load_seconds,
        "datasets": summaries,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
