#!/usr/bin/env python
"""Extract initial hidden states using the saved rollout prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dalsa.modeling import load_causal_lm, load_tokenizer
from dalsa.prompts import render_prompt
from dalsa.estimator import infer_num_rollouts, soft_fail_targets


@torch.no_grad()
def extract_batch(model, tokenizer, prompts, max_length):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-mode", choices=["saved", "shared", "native"], default="saved")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload["data"]
    rollout_data = {"metadata": payload.get("metadata", {}), "items": data}
    num_rollouts = infer_num_rollouts(rollout_data)
    soft_fail_targets(rollout_data)

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    model.eval()

    items = []
    for start in tqdm(range(0, len(data), args.batch_size), desc="extract h0"):
        batch = data[start : start + args.batch_size]
        prompts = []
        for item in batch:
            if args.prompt_mode == "saved":
                prompts.append(item["prompt"])
            else:
                prompts.append(render_prompt(tokenizer, args.model, item, args.prompt_mode))
        h0 = extract_batch(model, tokenizer, prompts, args.max_length)
        for item, prompt, vector in zip(batch, prompts, h0):
            items.append(
                {
                    "id": item["id"],
                    "h0": vector,
                    "is_hard": bool(item["is_hard"]),
                    "correct_count": int(item["correct_count"]),
                    "prompt": prompt,
                    "task_type": item.get("task_type"),
                }
            )

    result = {
        "metadata": {
            **payload.get("metadata", {}),
            "source": args.input,
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "positive_label": "hard",
            "num_rollouts": num_rollouts,
        },
        "hidden_dim": int(model.config.hidden_size),
        "items": items,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"saved {len(items)} h0 vectors to {output}")


if __name__ == "__main__":
    main()
