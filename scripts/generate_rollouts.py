#!/usr/bin/env python
"""Generate K rollouts and label each example as hard when not all rollouts are correct."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from dalsa.answers import SCORER_VERSION, is_correct
from dalsa.data import load_benchmark
from dalsa.modeling import generate_one, load_causal_lm, load_tokenizer, set_seed
from dalsa.prompts import render_prompt


def parse_years(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def shard_items(items, num_shards: int, shard_index: int):
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    if num_shards == 1:
        return items
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def generate_hf(args, items):
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    model.eval()

    results = []
    for item in tqdm(items, desc="HF rollouts"):
        prompt = render_prompt(tokenizer, args.model, item, args.prompt_mode)
        rollouts = []
        correct_count = 0
        for _ in range(args.num_rollouts):
            text, n_tokens = generate_one(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            ok, pred = is_correct(text, item)
            correct_count += int(ok)
            rollouts.append(
                {
                    "response": text,
                    "predicted_answer": pred,
                    "is_correct": ok,
                    "num_tokens": n_tokens,
                }
            )
        results.append(pack_result(item, prompt, rollouts, correct_count, args.num_rollouts))
    return results


def generate_vllm(args, items):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [render_prompt(tokenizer, args.model, item, args.prompt_mode) for item in items]
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
    sampling = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling)
    results = []
    for item, prompt, output in tqdm(list(zip(items, prompts, outputs)), desc="vLLM scoring"):
        rollouts = []
        correct_count = 0
        for sample in output.outputs:
            text = sample.text
            ok, pred = is_correct(text, item)
            correct_count += int(ok)
            rollouts.append(
                {
                    "response": text,
                    "predicted_answer": pred,
                    "is_correct": ok,
                    "num_tokens": len(sample.token_ids),
                }
            )
        results.append(pack_result(item, prompt, rollouts, correct_count, args.num_rollouts))
    return results


def pack_result(item, prompt, rollouts, correct_count, num_rollouts):
    result = dict(item)
    result.update(
        {
            "prompt": prompt,
            "rollouts": rollouts,
            "num_rollouts": num_rollouts,
            "correct_count": correct_count,
            "is_hard": correct_count < num_rollouts,
            "label_semantics": "hard=1 iff at least one rollout is wrong",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-mode", choices=["shared", "native"], default="shared")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aime-years", default=None, help="Comma-separated AIME years, e.g. 2024.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    all_items = load_benchmark(
        args.dataset,
        args.split,
        limit=args.limit,
        seed=args.seed,
        aime_years=parse_years(args.aime_years),
    )
    items = shard_items(all_items, args.num_shards, args.shard_index)
    results = generate_hf(args, items) if args.engine == "hf" else generate_vllm(args, items)

    total_correct = sum(item["correct_count"] for item in results)
    total_rollouts = len(results) * args.num_rollouts
    payload = {
        "metadata": {
            "scorer_version": SCORER_VERSION,
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "aime_years": parse_years(args.aime_years),
            "num_rollouts": args.num_rollouts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "positive_label": "hard",
            "accuracy": total_correct / total_rollouts if total_rollouts else 0.0,
            "hard_count": sum(int(item["is_hard"]) for item in results),
            "total": len(results),
            "unsharded_total": len(all_items),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
        "data": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
