"""Model loading and generation helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, device_map: str = "auto", dtype: str = "bfloat16"):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map=device_map,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )


@torch.no_grad()
def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    outputs = model.generate(**inputs, **generation_kwargs)
    gen_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, int(gen_ids.numel())
