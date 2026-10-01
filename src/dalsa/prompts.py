"""Prompt rendering policies for DALSA experiments."""

from __future__ import annotations

from typing import Any


def _choice_hint(item: dict[str, Any]) -> str:
    labels = item.get("choice_labels") or ["A", "B", "C", "D", "E"]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def build_user_prompt(item: dict[str, Any]) -> str:
    task_type = item.get("task_type", "math")
    if task_type == "mcq":
        choices = "\n".join(item["choices"])
        return (
            "Answer the following multiple choice question by reasoning step by step. "
            f"At the end, state your final answer as a single letter ({_choice_hint(item)}) "
            "inside \\boxed{}.\n\n"
            f"Question: {item['question']}\n"
            f"Choices:\n{choices}"
        )

    if task_type == "qa":
        context = item.get("context")
        if context:
            return (
                "Answer the following question using the provided context. "
                "Reason step by step, then provide a concise final answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {item['question']}"
            )
        return (
            "Answer the following question. Reason step by step, then provide a concise final answer.\n\n"
            f"Question: {item['question']}"
        )

    return (
        "Solve the following math problem by reasoning step by step. "
        "At the end, put the final answer inside \\boxed{}.\n\n"
        f"Problem: {item['question']}"
    )


def render_shared_prompt(item: dict[str, Any]) -> str:
    return build_user_prompt(item) + "\n\nSolution:"


def _model_is_deepseek_r1_distill(model_name: str) -> bool:
    lowered = model_name.lower()
    return "deepseek-r1-distill" in lowered or "deepseek_r1_distill" in lowered


def render_native_prompt(
    tokenizer: Any,
    model_name: str,
    item: dict[str, Any],
    force_deepseek_think: bool = True,
) -> str:
    user_prompt = build_user_prompt(item)
    messages = [{"role": "user", "content": user_prompt}]

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = render_shared_prompt(item)

    if force_deepseek_think and _model_is_deepseek_r1_distill(model_name):
        tail = prompt[-128:]
        if "<think>" not in tail:
            prompt += "<think>\n"
    return prompt


def render_prompt(
    tokenizer: Any,
    model_name: str,
    item: dict[str, Any],
    prompt_mode: str,
) -> str:
    if prompt_mode == "shared":
        return render_shared_prompt(item)
    if prompt_mode == "native":
        return render_native_prompt(tokenizer, model_name, item)
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")
