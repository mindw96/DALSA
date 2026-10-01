"""ACM-style model merging used by DALSA.

This module is intentionally explicit about lambda semantics:

    lambda_scale == amount of long-model delta injected into the short model.

The caller is responsible for mapping P(Hard | x) to lambda.
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ACMCalibrationInfo:
    short_model_name: str
    long_model_name: str
    tokenizer_name: str
    t_l: dict[int, float] = field(default_factory=dict)
    mi_l: dict[int, float] = field(default_factory=dict)
    layer_info: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    bias_info: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    layer_to_number: dict[str, int] = field(default_factory=dict)
    skipped_layers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokenizer: Any = None

    @property
    def t_x(self) -> dict[int, float]:
        """Backward-compatible alias for old calibration files/code."""
        return self.t_l

    @t_x.setter
    def t_x(self, value: dict[int, float]) -> None:
        self.t_l = value

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "short_model_name": self.short_model_name,
                "long_model_name": self.long_model_name,
                "tokenizer_name": self.tokenizer_name,
                "t_l": self.t_l,
                "mi_l": self.mi_l,
                "layer_info": self.layer_info,
                "bias_info": self.bias_info,
                "layer_to_number": self.layer_to_number,
                "skipped_layers": self.skipped_layers,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ACMCalibrationInfo":
        payload = torch.load(path, map_location="cpu")
        info = cls(
            short_model_name=payload["short_model_name"],
            long_model_name=payload["long_model_name"],
            tokenizer_name=payload["tokenizer_name"],
            t_l=payload.get("t_l", payload.get("t_x", {})),
            mi_l=payload.get("mi_l", {}),
            layer_info=payload["layer_info"],
            bias_info=payload["bias_info"],
            layer_to_number=payload["layer_to_number"],
            skipped_layers=payload.get("skipped_layers", []),
            metadata=payload.get("metadata", {}),
        )
        info.tokenizer = AutoTokenizer.from_pretrained(info.tokenizer_name, trust_remote_code=True)
        if info.tokenizer.pad_token is None:
            info.tokenizer.pad_token = info.tokenizer.eos_token
        return info


@dataclass
class FIMCalibrationInfo:
    short_model_name: str
    long_model_name: str
    tokenizer_name: str
    alpha_l: dict[int, float] = field(default_factory=dict)
    fim_l: dict[int, float] = field(default_factory=dict)
    delta_norm_l: dict[int, float] = field(default_factory=dict)
    risk_l: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokenizer: Any = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "short_model_name": self.short_model_name,
                "long_model_name": self.long_model_name,
                "tokenizer_name": self.tokenizer_name,
                "alpha_l": self.alpha_l,
                "fim_l": self.fim_l,
                "delta_norm_l": self.delta_norm_l,
                "risk_l": self.risk_l,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FIMCalibrationInfo":
        payload = torch.load(path, map_location="cpu")
        info = cls(
            short_model_name=payload["short_model_name"],
            long_model_name=payload["long_model_name"],
            tokenizer_name=payload["tokenizer_name"],
            alpha_l=payload["alpha_l"],
            fim_l=payload.get("fim_l", {}),
            delta_norm_l=payload.get("delta_norm_l", {}),
            risk_l=payload.get("risk_l", {}),
            metadata=payload.get("metadata", {}),
        )
        info.tokenizer = AutoTokenizer.from_pretrained(info.tokenizer_name, trust_remote_code=True)
        if info.tokenizer.pad_token is None:
            info.tokenizer.pad_token = info.tokenizer.eos_token
        return info


def _layer_number(name: str) -> int:
    if "embed" in name:
        return -1
    if name == "model.norm":
        return -2
    if "lm_head" in name:
        return -3
    found = re.findall(r"\d+", name)
    return int(found[0]) if found else -99


def _is_target_module(name: str) -> bool:
    return "down_proj" in name or "embed" in name or name == "model.norm" or "lm_head" in name


def load_calibration_texts(dataset_name: str, split: str, max_samples: int, seed: int) -> list[str]:
    local_path = Path(dataset_name)
    if local_path.exists():
        if local_path.suffix == ".jsonl":
            items = []
            with local_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        elif local_path.suffix == ".json":
            with local_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            items = payload.get("data", payload) if isinstance(payload, dict) else payload
        else:
            items = [{"text": line.strip()} for line in local_path.read_text(encoding="utf-8").splitlines()]

        texts = []
        for item in items:
            if isinstance(item, str):
                text = item
            else:
                text = (
                    item.get("text")
                    or item.get("prompt")
                    or item.get("question")
                    or item.get("problem")
                    or json.dumps(item, ensure_ascii=False)
                )
            text = str(text).strip()
            if text:
                texts.append(text)
        rng = random.Random(seed)
        rng.shuffle(texts)
        return texts[:max_samples] if max_samples > 0 else texts

    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)
    texts = []
    for item in ds:
        text = (
            item.get("question")
            or item.get("problem")
            or item.get("text")
            or item.get("sentence")
            or json.dumps(item, ensure_ascii=False)
        )
        text = str(text).strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def _cluster_rows(values: np.ndarray, max_clusters: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n_samples = values.shape[0]
    if n_samples < 2 or np.allclose(values, values[0]):
        return np.zeros(n_samples, dtype=np.int64)
    max_clusters = max(2, max_clusters)
    n_clusters = min(max_clusters, max(2, n_samples // 2))
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(values)


@torch.no_grad()
def _collect_features(model, tokenizer, texts: list[str], device: str, max_length: int) -> dict[str, list[torch.Tensor]]:
    features: dict[str, list[torch.Tensor]] = {}

    def hook(_module, _inputs, output, name: str):
        if isinstance(output, tuple):
            output = output[0]
        # Use max abs activation over tokens to keep calibration compact.
        value = output.detach().float().reshape(-1, output.shape[-1]).abs().amax(dim=0).cpu()
        features.setdefault(name, []).append(value)

    hooks = []
    for name, module in model.named_modules():
        if _is_target_module(name):
            hooks.append(module.register_forward_hook(lambda m, x, y, n=name: hook(m, x, y, n)))

    model = model.to(device)
    model.eval()
    for text in tqdm(texts, desc="calibration features"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        model(**inputs)

    for handle in hooks:
        handle.remove()
    return features


def precompute_acm_calibration(
    long_model_name: str,
    short_model_name: str,
    theta: float = 1.0,
    dataset_name: str = "simplescaling/s1K",
    split: str = "train",
    max_samples: int = 20,
    mi_clusters: int = 8,
    seed: int = 54,
    device: str = "cuda",
    max_length: int = 1024,
) -> ACMCalibrationInfo:
    tokenizer = AutoTokenizer.from_pretrained(short_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_calibration_texts(dataset_name, split, max_samples, seed)
    info = ACMCalibrationInfo(
        short_model_name=short_model_name,
        long_model_name=long_model_name,
        tokenizer_name=short_model_name,
        tokenizer=tokenizer,
        metadata={
            "calibration_dataset": dataset_name,
            "split": split,
            "max_samples": max_samples,
            "actual_samples": len(texts),
            "mi_clusters": mi_clusters,
            "theta": theta,
            "max_length": max_length,
            "target_modules": ["embed", "down_proj", "model.norm", "lm_head"],
            "t_l_semantics": "offline layer-wise consensus factor; runtime input only changes lambda",
        },
    )

    short_model = AutoModelForCausalLM.from_pretrained(short_model_name, torch_dtype=torch.bfloat16)
    long_model = AutoModelForCausalLM.from_pretrained(long_model_name, torch_dtype=torch.bfloat16)

    logger.info("Collecting short-model calibration features")
    short_features = _collect_features(short_model, tokenizer, texts, device, max_length)
    short_model = short_model.to("cpu")
    torch.cuda.empty_cache()

    logger.info("Collecting long-model calibration features")
    long_features = _collect_features(long_model, tokenizer, texts, device, max_length)
    long_model = long_model.to("cpu")
    torch.cuda.empty_cache()

    short_modules = dict(short_model.named_modules())
    long_modules = dict(long_model.named_modules())

    for name, long_module in tqdm(long_modules.items(), desc="merge deltas"):
        if not hasattr(long_module, "weight") or name not in short_modules:
            continue
        short_module = short_modules[name]
        if not hasattr(short_module, "weight"):
            continue
        if tuple(short_module.weight.shape) != tuple(long_module.weight.shape):
            info.skipped_layers.append(name)
            continue

        number = _layer_number(name)
        info.layer_to_number[name] = number
        base_weight = short_module.weight.data.detach().cpu().clone()
        delta = long_module.weight.data.detach().cpu() - base_weight
        info.layer_info[name] = {"base_weight": base_weight, "delta": delta}

        if (
            getattr(short_module, "bias", None) is not None
            and getattr(long_module, "bias", None) is not None
            and tuple(short_module.bias.shape) == tuple(long_module.bias.shape)
        ):
            base_bias = short_module.bias.data.detach().cpu().clone()
            bias_delta = long_module.bias.data.detach().cpu() - base_bias
            info.bias_info[name] = {"base_bias": base_bias, "delta": bias_delta}

    mi_by_module: dict[str, float] = {}
    for name in set(short_features) & set(long_features):
        short_matrix = torch.stack(short_features[name]).numpy()
        long_matrix = torch.stack(long_features[name]).numpy()
        labels_short = _cluster_rows(short_matrix, mi_clusters, seed)
        labels_long = _cluster_rows(long_matrix, mi_clusters, seed + 1)
        mi_by_module[name] = float(mutual_info_score(labels_short, labels_long))

    mi_by_layer: dict[int, list[float]] = {}
    for name, score in mi_by_module.items():
        mi_by_layer.setdefault(_layer_number(name), []).append(score)

    info.mi_l = {number: float(np.mean(scores)) for number, scores in mi_by_layer.items()}
    max_mi = max(info.mi_l.values()) if info.mi_l else 0.0
    info.metadata["max_mi"] = max_mi
    info.metadata["mi_estimator"] = "KMeans partitions over calibration samples"
    for number, mi in info.mi_l.items():
        info.t_l[number] = float(1.0 / (1.0 + np.exp(-theta * (mi - max_mi))))

    for number in set(info.layer_to_number.values()):
        info.t_l.setdefault(number, 0.0)

    del short_model, long_model
    torch.cuda.empty_cache()
    return info


@torch.no_grad()
def _compute_delta_norm_by_layer(short_model_name: str, long_model_name: str) -> dict[int, float]:
    short_model = _load_base_model_for_materialization(short_model_name)
    long_model = _load_base_model_for_materialization(long_model_name)
    long_params = dict(long_model.named_parameters())
    delta_norm_l: dict[int, float] = defaultdict(float)

    for name, short_param in tqdm(short_model.named_parameters(), desc="FIM delta norms"):
        long_param = long_params.get(name)
        if long_param is None or tuple(long_param.shape) != tuple(short_param.shape):
            continue
        number = _layer_number(name)
        short_tensor = short_param.detach().float()
        long_tensor = long_param.detach().float()
        delta_norm_l[number] += float((long_tensor - short_tensor).pow(2).sum().item())

    del short_model, long_model
    torch.cuda.empty_cache()
    return dict(delta_norm_l)


def _normalize_fim_coefficients(
    risk_l: dict[int, float],
    alpha_min: float,
    alpha_max: float,
    temperature: float,
) -> dict[int, float]:
    if not 0.0 <= alpha_min <= alpha_max <= 1.5:
        raise ValueError("--fim-alpha-min/max must satisfy 0 <= min <= max <= 1.5")
    if temperature <= 0:
        raise ValueError("--fim-temperature must be positive")

    positive_items = [(layer, value) for layer, value in risk_l.items() if value > 0]
    if not positive_items:
        return {layer: float(alpha_max) for layer in risk_l}

    layers = [layer for layer, _ in positive_items]
    log_scores = np.array([np.log(value + 1e-30) for _, value in positive_items], dtype=np.float64)
    center = float(np.mean(log_scores))
    scale = float(np.std(log_scores))
    if scale < 1e-8:
        normalized = np.zeros_like(log_scores)
    else:
        normalized = (log_scores - center) / scale

    # High FIM-risk layers receive lower merge coefficients.
    raw = 1.0 / (1.0 + np.exp(temperature * normalized))
    coeffs = alpha_min + (alpha_max - alpha_min) * raw
    return {layer: float(coeff) for layer, coeff in zip(layers, coeffs)}


def precompute_fim_calibration(
    long_model_name: str,
    short_model_name: str,
    *,
    random_samples: int = 8,
    random_seq_len: int = 128,
    random_batch_size: int = 1,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    temperature: float = 1.0,
    seed: int = 54,
    device: str = "cuda",
) -> FIMCalibrationInfo:
    """Approximate FIM-Merging layer coefficients from random-token gradients.

    This is a code-level reproduction path for the 2026 FIM-Merging preprint:
    estimate a diagonal FIM proxy on random token inputs, multiply it by the
    layer task-vector norm, then map log-risk scores to conservative/aggressive
    layer-wise merge coefficients.
    """

    if random_samples <= 0:
        raise ValueError("--fim-random-samples must be positive")
    if random_seq_len <= 1:
        raise ValueError("--fim-seq-len must be greater than 1")
    if random_batch_size <= 0:
        raise ValueError("--fim-batch-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(short_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        short_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    fim_sum_l: dict[int, float] = defaultdict(float)
    fim_count_l: dict[int, int] = defaultdict(int)

    remaining = random_samples
    progress = tqdm(total=random_samples, desc="FIM random-token gradients")
    while remaining > 0:
        batch_size = min(random_batch_size, remaining)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, random_seq_len),
            device=device,
            generator=generator,
        )
        attention_mask = torch.ones_like(input_ids)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        output.loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            number = _layer_number(name)
            grad = param.grad.detach().float()
            fim_sum_l[number] += float(grad.pow(2).sum().item())
            fim_count_l[number] += int(grad.numel())

        remaining -= batch_size
        progress.update(batch_size)

    progress.close()
    model.zero_grad(set_to_none=True)
    model = model.to("cpu")
    del model
    torch.cuda.empty_cache()

    fim_l = {
        number: float(fim_sum_l[number] / max(1, fim_count_l[number]))
        for number in fim_sum_l
    }
    delta_norm_l = _compute_delta_norm_by_layer(short_model_name, long_model_name)
    all_layers = set(fim_l) | set(delta_norm_l)
    risk_l = {
        number: float(fim_l.get(number, 0.0) * delta_norm_l.get(number, 0.0))
        for number in all_layers
    }
    alpha_l = _normalize_fim_coefficients(
        risk_l,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        temperature=temperature,
    )
    for number in all_layers:
        alpha_l.setdefault(number, float(alpha_max))

    return FIMCalibrationInfo(
        short_model_name=short_model_name,
        long_model_name=long_model_name,
        tokenizer_name=short_model_name,
        tokenizer=tokenizer,
        alpha_l=alpha_l,
        fim_l=fim_l,
        delta_norm_l=delta_norm_l,
        risk_l=risk_l,
        metadata={
            "calibration_type": "fim_random_tokens",
            "random_samples": random_samples,
            "random_seq_len": random_seq_len,
            "random_batch_size": random_batch_size,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "temperature": temperature,
            "seed": seed,
            "device": device,
            "coefficient_semantics": "theta_short + lambda * alpha_l * (theta_long - theta_short)",
        },
    )


def fast_acm_merge(
    calib_info: ACMCalibrationInfo,
    lambda_scale: float,
    base_model=None,
    device: str = "cuda",
    merge_base: str = "short",
    disable_layer_consensus: bool = False,
):
    if not 0.0 <= lambda_scale <= 1.5:
        raise ValueError(f"Suspicious lambda_scale={lambda_scale}; expected a small non-negative value.")
    if merge_base not in {"short", "long"}:
        raise ValueError("--merge-base must be one of: short, long")

    if base_model is None:
        model_name = calib_info.short_model_name if merge_base == "short" else calib_info.long_model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)

    with torch.no_grad():
        for name, param in base_model.named_parameters():
            layer_name = name.rsplit(".", 1)[0]
            if name.endswith(".weight") and layer_name in calib_info.layer_info:
                number = calib_info.layer_to_number[layer_name]
                consensus = 0.0 if disable_layer_consensus else float(calib_info.t_l.get(number, 0.0))
                scale_factor = (1.0 - consensus) * lambda_scale
                payload = calib_info.layer_info[layer_name]
                short_weight = payload["base_weight"].to(param.device, dtype=param.dtype)
                long_minus_short = payload["delta"].to(param.device, dtype=param.dtype)
                if merge_base == "short":
                    merged_weight = short_weight + long_minus_short * scale_factor
                else:
                    long_weight = short_weight + long_minus_short
                    merged_weight = long_weight - long_minus_short * scale_factor
                param.data.copy_(merged_weight)
            elif name.endswith(".bias") and layer_name in calib_info.bias_info:
                number = calib_info.layer_to_number[layer_name]
                consensus = 0.0 if disable_layer_consensus else float(calib_info.t_l.get(number, 0.0))
                scale_factor = (1.0 - consensus) * lambda_scale
                payload = calib_info.bias_info[layer_name]
                short_bias = payload["base_bias"].to(param.device, dtype=param.dtype)
                long_minus_short = payload["delta"].to(param.device, dtype=param.dtype)
                if merge_base == "short":
                    merged_bias = short_bias + long_minus_short * scale_factor
                else:
                    long_bias = short_bias + long_minus_short
                    merged_bias = long_bias - long_minus_short * scale_factor
                param.data.copy_(merged_bias)

    return base_model, calib_info.tokenizer


def _load_base_model_for_materialization(model_name: str):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs.pop("low_cpu_mem_usage", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def materialize_full_task_arithmetic_model(
    short_model_name: str,
    long_model_name: str,
    lambda_scale: float,
    output_dir: str | Path,
    *,
    force: bool = False,
    max_shard_size: str = "5GB",
    merge_base: str = "short",
) -> Path:
    """Write a full-parameter task-arithmetic checkpoint loadable by vLLM."""

    if not 0.0 <= lambda_scale <= 1.5:
        raise ValueError(f"Suspicious lambda_scale={lambda_scale}; expected a small non-negative value.")
    if merge_base not in {"short", "long"}:
        raise ValueError("--merge-base must be one of: short, long")

    output_dir = Path(output_dir)
    marker = output_dir / "dalsa_merged_metadata.json"
    if marker.exists() and (output_dir / "config.json").exists() and not force:
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = short_model_name if merge_base == "short" else long_model_name
    other_model_name = long_model_name if merge_base == "short" else short_model_name
    model = _load_base_model_for_materialization(base_model_name)
    other_model = _load_base_model_for_materialization(other_model_name)
    other_params = dict(other_model.named_parameters())

    merged_count = 0
    skipped = []
    with torch.no_grad():
        for name, param in tqdm(model.named_parameters(), desc="full TA merge"):
            other_param = other_params.get(name)
            if other_param is None or tuple(other_param.shape) != tuple(param.shape):
                skipped.append(name)
                continue
            other_tensor = other_param.detach().to(param.device, dtype=param.dtype)
            merged = param.detach() + (other_tensor - param.detach()) * lambda_scale
            param.data.copy_(merged)
            merged_count += 1

    tokenizer = AutoTokenizer.from_pretrained(short_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                "short_model_name": short_model_name,
                "long_model_name": long_model_name,
                "tokenizer_name": short_model_name,
                "lambda_scale": float(lambda_scale),
                "merge_base": merge_base,
                "merge_scope": "full_model",
                "merged_parameter_count": merged_count,
                "skipped_parameter_count": len(skipped),
                "skipped_parameters": skipped[:100],
                "source": "materialize_full_task_arithmetic_model",
                "semantics": (
                    "theta_short + lambda_scale * (theta_long - theta_short)"
                    if merge_base == "short"
                    else "theta_long + lambda_scale * (theta_short - theta_long)"
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    del model, other_model
    torch.cuda.empty_cache()
    return output_dir


def materialize_sparse_task_arithmetic_model(
    short_model_name: str,
    long_model_name: str,
    lambda_scale: float,
    output_dir: str | Path,
    *,
    mode: str,
    ties_density: float = 0.2,
    dare_drop_rate: float = 0.5,
    seed: int = 42,
    force: bool = False,
    max_shard_size: str = "5GB",
    merge_base: str = "short",
) -> Path:
    """Write a sparsified task-arithmetic checkpoint loadable by vLLM.

    mode="ties": per-tensor magnitude trimming that keeps the top
    ``ties_density`` fraction of delta entries. With a single task vector the
    TIES sign-election and disjoint-merge steps are identities, so trimming is
    the full TIES procedure in this two-model setting.
    mode="dare": per-tensor Bernoulli drop of ``dare_drop_rate`` of the delta
    entries with 1/(1-p) rescaling of the survivors.
    """

    if not 0.0 <= lambda_scale <= 1.5:
        raise ValueError(f"Suspicious lambda_scale={lambda_scale}; expected a small non-negative value.")
    if merge_base not in {"short", "long"}:
        raise ValueError("--merge-base must be one of: short, long")
    if mode not in {"ties", "dare"}:
        raise ValueError("mode must be one of: ties, dare")
    if mode == "ties" and not 0.0 < ties_density <= 1.0:
        raise ValueError("--ties-density must be in (0, 1]")
    if mode == "dare" and not 0.0 <= dare_drop_rate < 1.0:
        raise ValueError("--dare-drop-rate must be in [0, 1)")

    output_dir = Path(output_dir)
    marker = output_dir / "dalsa_merged_metadata.json"
    if marker.exists() and (output_dir / "config.json").exists() and not force:
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = short_model_name if merge_base == "short" else long_model_name
    other_model_name = long_model_name if merge_base == "short" else short_model_name
    model = _load_base_model_for_materialization(base_model_name)
    other_model = _load_base_model_for_materialization(other_model_name)
    other_params = dict(other_model.named_parameters())

    merged_count = 0
    skipped = []
    kept_entries = 0
    total_entries = 0
    with torch.no_grad():
        for name, param in tqdm(model.named_parameters(), desc=f"{mode.upper()} sparse TA merge"):
            other_param = other_params.get(name)
            if other_param is None or tuple(other_param.shape) != tuple(param.shape):
                skipped.append(name)
                continue
            other_tensor = other_param.detach().to(param.device, dtype=param.dtype)
            delta = (other_tensor - param.detach()).float()
            if mode == "ties":
                if ties_density < 1.0 and delta.numel() > 1:
                    magnitude = delta.abs().flatten()
                    keep = max(1, int(round(ties_density * magnitude.numel())))
                    threshold = torch.topk(magnitude, keep, largest=True).values[-1]
                    mask = (delta.abs() >= threshold).to(delta.dtype)
                    delta = delta * mask
                    kept_entries += int(mask.sum().item())
                else:
                    kept_entries += delta.numel()
            else:
                tensor_seed = (seed * 1000003 + _stable_name_hash(name)) % (2**63 - 1)
                generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
                mask = (
                    torch.rand(delta.shape, generator=generator, device="cpu") >= dare_drop_rate
                ).to(delta.device, dtype=delta.dtype)
                delta = delta * mask / max(1e-8, 1.0 - dare_drop_rate)
                kept_entries += int(mask.sum().item())
            total_entries += delta.numel()
            merged = param.detach().float() + delta * lambda_scale
            param.data.copy_(merged.to(param.dtype))
            merged_count += 1

    tokenizer = AutoTokenizer.from_pretrained(short_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                "short_model_name": short_model_name,
                "long_model_name": long_model_name,
                "tokenizer_name": short_model_name,
                "lambda_scale": float(lambda_scale),
                "merge_base": merge_base,
                "merge_scope": f"{mode}_full",
                "ties_density": float(ties_density) if mode == "ties" else None,
                "dare_drop_rate": float(dare_drop_rate) if mode == "dare" else None,
                "seed": int(seed),
                "merged_parameter_count": merged_count,
                "skipped_parameter_count": len(skipped),
                "skipped_parameters": skipped[:100],
                "kept_delta_fraction": (kept_entries / total_entries) if total_entries else None,
                "source": "materialize_sparse_task_arithmetic_model",
                "semantics": (
                    f"theta_short + lambda_scale * sparsify_{mode}(theta_long - theta_short)"
                    if merge_base == "short"
                    else f"theta_long + lambda_scale * sparsify_{mode}(theta_short - theta_long)"
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    del model, other_model
    torch.cuda.empty_cache()
    return output_dir


def _stable_name_hash(name: str) -> int:
    value = 0
    for ch in name:
        value = (value * 131 + ord(ch)) % (2**61 - 1)
    return value


def materialize_fim_merged_model(
    fim_info: FIMCalibrationInfo,
    lambda_scale: float,
    output_dir: str | Path,
    *,
    force: bool = False,
    max_shard_size: str = "5GB",
    merge_base: str = "short",
) -> Path:
    """Write a FIM-weighted task-arithmetic checkpoint loadable by vLLM."""

    if not 0.0 <= lambda_scale <= 1.5:
        raise ValueError(f"Suspicious lambda_scale={lambda_scale}; expected a small non-negative value.")
    if merge_base not in {"short", "long"}:
        raise ValueError("--merge-base must be one of: short, long")

    output_dir = Path(output_dir)
    marker = output_dir / "dalsa_merged_metadata.json"
    if marker.exists() and (output_dir / "config.json").exists() and not force:
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = fim_info.short_model_name if merge_base == "short" else fim_info.long_model_name
    other_model_name = fim_info.long_model_name if merge_base == "short" else fim_info.short_model_name
    model = _load_base_model_for_materialization(base_model_name)
    other_model = _load_base_model_for_materialization(other_model_name)
    other_params = dict(other_model.named_parameters())

    merged_count = 0
    skipped = []
    used_alpha: dict[int, float] = {}
    with torch.no_grad():
        for name, param in tqdm(model.named_parameters(), desc="FIM-weighted TA merge"):
            other_param = other_params.get(name)
            if other_param is None or tuple(other_param.shape) != tuple(param.shape):
                skipped.append(name)
                continue
            number = _layer_number(name)
            alpha = float(fim_info.alpha_l.get(number, 1.0))
            used_alpha[number] = alpha
            other_tensor = other_param.detach().to(param.device, dtype=param.dtype)
            scale_factor = lambda_scale * alpha
            merged = param.detach() + (other_tensor - param.detach()) * scale_factor
            param.data.copy_(merged)
            merged_count += 1

    tokenizer = AutoTokenizer.from_pretrained(fim_info.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                "short_model_name": fim_info.short_model_name,
                "long_model_name": fim_info.long_model_name,
                "tokenizer_name": fim_info.tokenizer_name,
                "lambda_scale": float(lambda_scale),
                "merge_base": merge_base,
                "merge_scope": "fim_full",
                "merged_parameter_count": merged_count,
                "skipped_parameter_count": len(skipped),
                "skipped_parameters": skipped[:100],
                "fim_metadata": fim_info.metadata,
                "alpha_by_layer": {str(k): v for k, v in sorted(used_alpha.items())},
                "source": "materialize_fim_merged_model",
                "semantics": (
                    "theta_short + lambda_scale * alpha_l * (theta_long - theta_short)"
                    if merge_base == "short"
                    else "theta_long + lambda_scale * alpha_l * (theta_short - theta_long)"
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    del model, other_model
    torch.cuda.empty_cache()
    return output_dir


def materialize_acm_merged_model(
    calib_info: ACMCalibrationInfo,
    lambda_scale: float,
    output_dir: str | Path,
    *,
    force: bool = False,
    max_shard_size: str = "5GB",
    merge_base: str = "short",
    disable_layer_consensus: bool = False,
) -> Path:
    """Write a merged ACM checkpoint that can be loaded by vLLM.

    vLLM cannot change full model weights per request. For fixed-merge or
    bucketed DALSA evaluation, we materialize one short+delta checkpoint per
    lambda value and then batch all examples assigned to that value.
    """

    output_dir = Path(output_dir)
    marker = output_dir / "dalsa_merged_metadata.json"
    if marker.exists() and (output_dir / "config.json").exists() and not force:
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge_base not in {"short", "long"}:
        raise ValueError("--merge-base must be one of: short, long")
    model_name = calib_info.short_model_name if merge_base == "short" else calib_info.long_model_name
    model = _load_base_model_for_materialization(model_name)
    model, tokenizer = fast_acm_merge(
        calib_info,
        lambda_scale=lambda_scale,
        base_model=model,
        device="cpu",
        merge_base=merge_base,
        disable_layer_consensus=disable_layer_consensus,
    )
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                "short_model_name": calib_info.short_model_name,
                "long_model_name": calib_info.long_model_name,
                "tokenizer_name": calib_info.tokenizer_name,
                "lambda_scale": float(lambda_scale),
                "merge_base": merge_base,
                "disable_layer_consensus": bool(disable_layer_consensus),
                "source": "materialize_acm_merged_model",
                "semantics": (
                    "theta_short + lambda_scale * (theta_long - theta_short)"
                    if disable_layer_consensus and merge_base == "short"
                    else "theta_long + lambda_scale * (theta_short - theta_long)"
                    if disable_layer_consensus
                    else "theta_short + (1 - t_l) * lambda_scale * (theta_long - theta_short)"
                    if merge_base == "short"
                    else "theta_long + (1 - t_l) * lambda_scale * (theta_short - theta_long)"
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    del model
    torch.cuda.empty_cache()
    return output_dir


def fast_acm_merge_state_dict_only(calib_info: ACMCalibrationInfo, lambda_scale: float) -> dict[str, torch.Tensor]:
    state_dict = {}
    for name, payload in calib_info.layer_info.items():
        number = calib_info.layer_to_number[name]
        scale_factor = (1.0 - float(calib_info.t_l.get(number, 0.0))) * lambda_scale
        state_dict[f"{name}.weight"] = (
            payload["base_weight"] + payload["delta"] * scale_factor
        ).to(torch.bfloat16)
    for name, payload in calib_info.bias_info.items():
        number = calib_info.layer_to_number[name]
        scale_factor = (1.0 - float(calib_info.t_l.get(number, 0.0))) * lambda_scale
        state_dict[f"{name}.bias"] = (
            payload["base_bias"] + payload["delta"] * scale_factor
        ).to(torch.bfloat16)
    return state_dict
