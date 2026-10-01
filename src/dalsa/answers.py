"""Answer extraction and correctness checks."""

from __future__ import annotations

import math
import re
import string
from typing import Any, Optional

import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

SCORER_VERSION = "dalsa-v2"


def extract_boxed(text: str) -> Optional[str]:
    marker = r"\boxed{"
    search_end = len(text)
    while True:
        start = text.rfind(marker, 0, search_end)
        if start < 0:
            return None

        i = start + len(marker)
        depth = 1
        out = []
        while i < len(text) and depth:
            ch = text[i]
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth:
                    out.append(ch)
            else:
                out.append(ch)
            i += 1

        if not depth:
            return "".join(out).strip()
        search_end = start


def extract_number(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    elif "####" in text:
        text = text.split("####")[-1]

    text = text.replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def extract_math_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    if "####" in text:
        return text.split("####")[-1].strip()
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return extract_number(text)


def extract_mcq(text: str, labels: str = "ABCDE") -> Optional[str]:
    boxed = extract_boxed(text)
    search_text = boxed if boxed else text
    labels = "".join(sorted(set(labels.upper())))

    direct = search_text.strip().upper()
    if direct in labels:
        return direct
    direct_match = re.match(rf"^\s*([{labels}])(?:\s*[\).:]|\s*$)", direct)
    if direct_match:
        return direct_match.group(1)

    patterns = [
        rf"(?:answer|choice|option|letter)\s*(?:is|:)?\s*\(?\s*([{labels}])\s*\)?",
        rf"(?:final)\s*(?:answer|choice|option)?\s*(?:is|:)?\s*\(?\s*([{labels}])\s*\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, search_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    tail = search_text.strip()[-200:]
    matches = re.findall(rf"(?:^|[\s\(\[])([{labels}])(?:[\s\)\].,;:]|$)", tail)
    if matches:
        return matches[-1].upper()
    return None


def _strip_math_wrappers(text: Any) -> str:
    text = str(text).strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    text = text.replace("\\$", "")
    text = text.replace("\\%", "%")
    text = text.replace("$", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\\s+", "", text)
    text = re.sub(r"\\(?:,|!|;|:)", "", text)
    text = re.sub(r"\^\s*\\circ|\\circ|°", "", text)
    text = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", text)
    return text.strip().rstrip(".")


def _remove_math_units(text: str) -> str:
    text = re.sub(
        r"\b(?:degree|degrees|inch|inches|foot|feet|meter|meters|centimeter|centimeters|cm|mile|miles|gallon|gallons|cent|cents|second|seconds|unit|units|square|cubic)\b(?:\s*\^\s*\d+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bper\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_frac(text: str) -> str:
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\frac\s*([-+]?\d+)\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\s*([-+]?\d+)", r"(\1)/(\2)", text)
    text = re.sub(r"\\frac\s*([-+]?\d+)\s*([-+]?\d+)", r"(\1)/(\2)", text)
    return text


def _strip_simple_assignment(text: str) -> str:
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text.strip())
    return re.sub(r"^[a-zA-Z](?:\s*\\in\b|\s+in\b)\s*", "", text)


def normalize_math_answer(text: Any) -> str:
    text = _strip_math_wrappers(text).lower()
    text = _strip_simple_assignment(text)
    text = _remove_math_units(text)
    text = _normalize_frac(text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("-(", "(-")
    text = re.sub(r"(?<![\w)])([-+]?\d+)/([-+]?\d+)", r"(\1)/(\2)", text)
    text = text.replace("{", "").replace("}", "")
    if re.fullmatch(r"\(?[a-e]\)?", text):
        text = text.strip("()")
    return text


def _numeric_value(text: Any) -> Optional[float]:
    raw = _strip_math_wrappers(text)
    raw = _strip_simple_assignment(raw)
    raw = _remove_math_units(raw)
    raw = raw.replace(",", "")
    raw = _normalize_frac(raw)

    frac = re.fullmatch(r"\\frac\{([-+]?\d*\.?\d+)\}\{([-+]?\d*\.?\d+)\}", raw)
    if frac and float(frac.group(2)) != 0.0:
        return float(frac.group(1)) / float(frac.group(2))

    simple_frac = re.fullmatch(r"\(?([-+]?\d*\.?\d+)\)?\s*/\s*\(?([-+]?\d*\.?\d+)\)?", raw)
    if simple_frac and float(simple_frac.group(2)) != 0.0:
        return float(simple_frac.group(1)) / float(simple_frac.group(2))

    sqrt = re.fullmatch(r"\\sqrt\{([-+]?\d*\.?\d+)\}", raw)
    if sqrt:
        radicand = float(sqrt.group(1))
        return math.sqrt(radicand) if radicand >= 0 else None

    if raw.endswith("%"):
        try:
            return float(raw[:-1])
        except ValueError:
            return None

    compact = re.sub(r"\s+", "", raw)
    try:
        return float(compact)
    except ValueError:
        pass

    return None


def _list_parts(text: Any) -> Optional[list[str]]:
    raw = _strip_math_wrappers(text)
    raw = re.sub(r",\s*\\!", "", raw)
    if "," not in raw:
        return None
    compact = re.sub(r"\s+", "", raw)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", compact):
        return None
    parts = [normalize_math_answer(part) for part in raw.split(",")]
    parts = [part for part in parts if part]
    return sorted(parts) if len(parts) > 1 else None


def _sympy_expr(text: Any) -> Optional[sp.Expr]:
    raw = _strip_math_wrappers(text).lower()
    if r"\infty" in raw:
        return sp.nan
    if any(marker in raw for marker in ("\\begin", "\\end", "\\pm", "_", "\\in", "\\sqrt[")):
        return None
    raw = _strip_simple_assignment(raw)
    raw = _remove_math_units(raw)
    raw = _normalize_frac(raw)
    raw = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", raw)
    raw = re.sub(r"\\sqrt\s*([a-z0-9]+)", r"sqrt(\1)", raw)
    raw = raw.replace("^", "**")
    raw = raw.replace("\\", "")
    raw = raw.replace("{", "(").replace("}", ")")
    raw = raw.replace(",", "")
    raw = raw.strip()
    if not raw or re.search(r"[^0-9a-zA-Z_+\-*/(). ]", raw):
        return None
    functions = {name: getattr(sp, name) for name in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "exp")}
    functions["abs"] = sp.Abs
    # ponytail: cap symbolic grading; extend only for a verified unsupported answer.
    powers = re.findall(r"\*\*\s*([+-]?\d+)\b", raw)
    if (len(raw) > 256 or raw.count("**") != len(powers)
            or math.prod(max(1, abs(int(power))) for power in powers) > 1000
            or re.search(r"\*\*\s*[+-]?\d+\s*\*\*", raw)
            or re.search(r"[a-z)]\s*\.", raw)
            or any(len(name) > 1 and name not in functions for name in re.findall(r"([a-z]+)\s*\(", raw))):
        return sp.nan
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        symbols = {name: getattr(sp, name) for name in ("Symbol", "Function", "Integer", "Float", "Rational", "Add", "Mul", "Pow")}
        expr = parse_expr(raw, transformations=transformations, evaluate=True,
                          local_dict={**functions, "pi": sp.pi, "oo": sp.oo, "nan": sp.nan, "zoo": sp.zoo},
                          global_dict={"__builtins__": {}, **symbols})
        return expr if isinstance(expr, sp.Expr) else sp.nan
    except Exception:
        return None


def normalize_text(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    text = text.lower().translate(table)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_short_qa(text: str) -> str:
    text = text.strip()
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    first_sentence = first_line.split(".")[0].strip()
    return first_line if len(first_line) <= 120 else first_sentence


def same_number(pred: Optional[str], gold: Any) -> bool:
    if pred is None:
        return False
    gold_str = str(gold).replace(",", "").strip()
    pred_str = str(pred).replace(",", "").strip()
    if pred_str == gold_str:
        return True
    try:
        return math.isclose(float(pred_str), float(gold_str), rel_tol=0.0, abs_tol=1e-5)
    except ValueError:
        return False


def same_math(pred: Optional[str], gold: Any) -> bool:
    if pred is None:
        return False
    pred_parts = _list_parts(pred)
    gold_parts = _list_parts(gold)
    if pred_parts is not None or gold_parts is not None:
        return (pred_parts is not None and gold_parts is not None
                and len(pred_parts) == len(gold_parts)
                and all(same_math(p, g) for p, g in zip(pred_parts, gold_parts)))
    pred_num = _numeric_value(pred)
    gold_num = _numeric_value(gold)
    if any(value is not None and not math.isfinite(value) for value in (pred_num, gold_num)):
        return False
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=0.0, abs_tol=1e-5)
    pred_expr = _sympy_expr(pred)
    gold_expr = _sympy_expr(gold)
    if any(expr is not None and expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for expr in (pred_expr, gold_expr)):
        return False
    if pred_expr is None or gold_expr is None:
        pred_norm = normalize_math_answer(pred)
        gold_norm = normalize_math_answer(gold)
        return bool(pred_norm and gold_norm and pred_norm == gold_norm)
    try:
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def same_aime(pred: Optional[str], gold: Any) -> bool:
    if pred is None:
        return False
    pred_digits = _strip_math_wrappers(pred)
    gold_digits = _strip_math_wrappers(gold)
    if not re.fullmatch(r"[0-9]+", pred_digits) or not re.fullmatch(r"[0-9]+", gold_digits):
        return False
    pred_digits = pred_digits.lstrip("0") or "0"
    gold_digits = gold_digits.lstrip("0") or "0"
    return len(pred_digits) <= 3 and len(gold_digits) <= 3 and pred_digits == gold_digits


def is_correct(response: str, item: dict[str, Any]) -> tuple[bool, Optional[str]]:
    task_type = item.get("task_type", "math")
    if task_type == "mcq":
        labels = "".join(item.get("choice_labels", ["A", "B", "C", "D", "E"]))
        pred = extract_mcq(response, labels=labels)
        return pred == str(item["ground_truth"]).upper(), pred

    if task_type == "qa":
        pred = extract_short_qa(response)
        pred_norm = normalize_text(pred)
        gold_norm = normalize_text(str(item["ground_truth"]))
        ok = pred_norm == gold_norm or gold_norm in pred_norm
        return ok, pred

    pred = extract_math_answer(response)
    if task_type == "aime":
        return same_aime(pred, item["ground_truth"]), pred
    return same_math(pred, item["ground_truth"]), pred
