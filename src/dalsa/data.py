"""Dataset loaders normalized to a shared DALSA schema."""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from datasets import load_dataset


def _limit(items: list[dict[str, Any]], limit: Optional[int], seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [items[i] for i in keep]


def _boxed_from_solution(solution: str) -> str:
    idx = solution.rfind(r"\boxed")
    if idx < 0:
        return ""
    start = solution.find("{", idx)
    if start < 0:
        return ""
    depth = 0
    for i, ch in enumerate(solution[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : i].strip()
    return ""


def _first_present(item: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def load_gsm8k(split: str) -> list[dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for idx, item in enumerate(ds):
        match = re.search(r"####\s*(.+?)$", item["answer"], flags=re.MULTILINE)
        answer = match.group(1).replace(",", "").strip() if match else ""
        rows.append(
            {
                "id": f"gsm8k_{split}_{idx}",
                "question": item["question"],
                "ground_truth": answer,
                "task_type": "math",
            }
        )
    return rows


def load_math500(split: str = "test", source: str | None = None) -> list[dict[str, Any]]:
    if source is None:
        source = "HuggingFaceH4/MATH-500" if split == "test" else "ricdomolm/MATH-500"
    ds = load_dataset(source, split=split)
    rows = []
    for idx, item in enumerate(ds):
        question = _first_present(item, ("problem", "Problem", "question", "Question", "prompt"))
        solution = str(_first_present(item, ("solution", "Solution", "explanation"), ""))
        answer = _first_present(
            item,
            ("answer", "Answer", "ground_truth", "final_answer", "label", "target"),
            None,
        )
        if answer is None or answer == "":
            answer = _boxed_from_solution(solution) or solution
        rows.append(
            {
                "id": str(_first_present(item, ("id", "ID", "problem_id"), f"math500_{split}_{idx}")),
                "question": str(question),
                "ground_truth": str(answer),
                "solution": solution,
                "subject": _first_present(item, ("subject", "Subject"), ""),
                "level": _first_present(item, ("level", "Level"), 0),
                "source_dataset": source,
                "task_type": "math",
            }
        )
    return rows


def load_competition_math(split: str, exclude_math500: bool = True) -> list[dict[str, Any]]:
    ds = load_dataset("qwedsacf/competition_math", split=split)
    math500 = set()
    if exclude_math500:
        for item in load_dataset("HuggingFaceH4/MATH-500", split="test"):
            math500.add(" ".join(item["problem"].split()))

    rows = []
    for idx, item in enumerate(ds):
        norm_problem = " ".join(item["problem"].split())
        if norm_problem in math500:
            continue
        rows.append(
            {
                "id": f"competition_math_{split}_{idx}",
                "question": item["problem"],
                "ground_truth": _boxed_from_solution(item.get("solution", "")),
                "solution": item.get("solution", ""),
                "task_type": "math",
            }
        )
    return rows


def load_aime2024() -> list[dict[str, Any]]:
    return load_math_ai_aime(2024)


def load_math_ai_aime(year: int, split: str = "test") -> list[dict[str, Any]]:
    if year < 2000:
        year += 2000
    if year not in {2024, 2025, 2026}:
        raise ValueError(f"math-ai AIME loader supports 2024, 2025, and 2026 only, got {year}.")

    suffix = str(year)[-2:]
    dataset_name = f"math-ai/aime{suffix}"
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for idx, item in enumerate(ds):
        source_id = _first_present(item, ("id", "ID", "problem_id", "Problem ID", "Problem Number"), idx)
        question = _first_present(item, ("problem", "Problem", "question", "Question", "prompt"))
        solution = str(_first_present(item, ("solution", "Solution", "explanation"), ""))
        answer = _first_present(
            item,
            ("answer", "Answer", "ground_truth", "final_answer", "label", "target"),
            None,
        )
        if answer is None or answer == "":
            answer = _boxed_from_solution(solution) or solution
        rows.append(
            {
                "id": f"aime_{year}_{source_id}",
                "question": str(question),
                "ground_truth": str(answer),
                "solution": solution,
                "year": year,
                "problem_id": source_id,
                "source_dataset": dataset_name,
                "task_type": "aime",
            }
        )
    return rows


def load_math_ai_amc23(split: str = "test") -> list[dict[str, Any]]:
    dataset_name = "math-ai/amc23"
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for idx, item in enumerate(ds):
        source_id = _first_present(item, ("id", "ID", "problem_id", "Problem ID"), idx)
        question = _first_present(item, ("question", "Question", "problem", "Problem", "prompt"))
        answer = _first_present(
            item,
            ("answer", "Answer", "ground_truth", "final_answer", "label", "target"),
            "",
        )
        rows.append(
            {
                "id": f"amc23_{source_id}",
                "question": str(question),
                "ground_truth": str(answer),
                "problem_id": source_id,
                "url": _first_present(item, ("url", "URL", "source"), ""),
                "source_dataset": dataset_name,
                "task_type": "amc",
            }
        )
    return rows


def load_aime(split: str, years: Optional[list[int]] = None) -> list[dict[str, Any]]:
    rows = []
    if years is None:
        years = [2024] if split == "test" else [2020, 2021, 2022, 2023]

    math_ai_years = [year for year in years if year in {2024, 2025, 2026}]
    for year in math_ai_years:
        rows.extend(load_math_ai_aime(year, split=split))

    years = [year for year in years if year not in {2024, 2025, 2026}]
    if not years:
        return rows

    if any(year > 2026 for year in years):
        raise ValueError("Use supported math-ai AIME datasets for years after 2026.")

    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    for idx, item in enumerate(ds):
        year = item.get("Year", item.get("year"))
        if year not in years:
            continue
        question = item.get("Problem") or item.get("problem") or item.get("Question") or item.get("question", "")
        source_id = item.get("ID") or item.get("Problem ID") or item.get("Problem Number", idx)
        problem_id = item.get("Problem Number", item.get("Problem ID", idx))
        rows.append(
            {
                "id": f"aime_{source_id}",
                "question": question,
                "ground_truth": str(item.get("Answer", item.get("answer", ""))),
                "year": year,
                "problem_id": problem_id,
                "part": item.get("Part", ""),
                "task_type": "aime",
            }
        )
    return rows


def _mcq_rows(ds, name: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(ds):
        labels = list(item["choices"]["label"])
        texts = list(item["choices"]["text"])
        default_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(texts)])
        if labels and all(label.isdigit() for label in labels):
            labels = default_labels[: len(labels)]
        choices = [f"{label}. {text}" for label, text in zip(labels, texts)]
        answer = str(item["answerKey"])
        if answer.isdigit():
            answer_index = int(answer) - 1
            if not 0 <= answer_index < len(default_labels):
                raise ValueError(f"Invalid numeric answerKey={answer!r} for {name} item {idx}.")
            answer = default_labels[answer_index]
        rows.append(
            {
                "id": item.get("id", f"{name}_{split}_{idx}"),
                "question": item["question"],
                "choices": choices,
                "choice_labels": labels,
                "choice_texts": texts,
                "ground_truth": answer,
                "task_type": "mcq",
            }
        )
    return rows


def load_arc(split: str) -> list[dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    return _mcq_rows(ds, "arc", split)


def load_csqa(split: str) -> list[dict[str, Any]]:
    if split == "test":
        split = "validation"
    ds = load_dataset("tau/commonsense_qa", split=split)
    rows = []
    for idx, item in enumerate(ds):
        labels = list(item["choices"]["label"])
        texts = list(item["choices"]["text"])
        rows.append(
            {
                "id": item.get("id", f"csqa_{split}_{idx}"),
                "question": item["question"],
                "choices": [f"{label}. {text}" for label, text in zip(labels, texts)],
                "choice_labels": labels,
                "choice_texts": texts,
                "ground_truth": item.get("answerKey", ""),
                "question_concept": item.get("question_concept", ""),
                "task_type": "mcq",
            }
        )
    return rows


def load_hotpotqa(split: str, include_context: bool = True) -> list[dict[str, Any]]:
    if split == "test":
        split = "validation"
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    rows = []
    for idx, item in enumerate(ds):
        row = {
            "id": item.get("id", f"hotpotqa_{split}_{idx}"),
            "question": item["question"],
            "ground_truth": item["answer"],
            "question_type": item.get("type", ""),
            "level": item.get("level", ""),
            "task_type": "qa",
        }
        if include_context:
            paragraphs = []
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                paragraphs.append(f"[{title}]\n{' '.join(sentences)}")
            row["context"] = "\n\n".join(paragraphs)
        rows.append(row)
    return rows


def load_benchmark(
    name: str,
    split: str,
    limit: Optional[int] = None,
    seed: int = 42,
    aime_years: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    key = name.lower()
    if key == "gsm8k":
        rows = load_gsm8k(split)
    elif key in {"math", "competition_math"}:
        rows = load_competition_math(split)
    elif key in {"math500", "math-500"}:
        rows = load_math500(split)
    elif key in {"ricdomolm/math-500", "ricdomolm_math500"}:
        rows = load_math500(split, source="ricdomolm/MATH-500")
    elif key in {"huggingfaceh4/math-500", "h4_math500"}:
        rows = load_math500(split, source="HuggingFaceH4/MATH-500")
    elif key == "aime":
        rows = load_aime(split, years=aime_years)
    elif key in {"aime24", "aime_2024", "math-ai/aime24"}:
        rows = load_aime(split, years=[2024])
    elif key in {"aime25", "aime_2025", "math-ai/aime25"}:
        rows = load_aime(split, years=[2025])
    elif key in {"aime26", "aime_2026", "math-ai/aime26"}:
        rows = load_aime(split, years=[2026])
    elif key in {"amc23", "amc_2023", "math-ai/amc23"}:
        rows = load_math_ai_amc23(split)
    elif key in {"arc", "arc_challenge"}:
        rows = load_arc(split)
    elif key in {"csqa", "commonsense_qa"}:
        rows = load_csqa(split)
    elif key in {"hotpotqa", "hotpot_qa"}:
        rows = load_hotpotqa(split)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return _limit(rows, limit, seed)
