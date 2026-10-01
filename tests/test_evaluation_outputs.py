import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dalsa.answers import SCORER_VERSION


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load_evaluator("evaluate_vllm")
adaptive = load_evaluator("evaluate_adaptive_vllm")


class EvaluationOutputTests(unittest.TestCase):
    def test_slash_aliases_write_local_files_with_scorer_provenance(self):
        dataset, split = "math-ai/aime24", "test/part"
        item = {"id": "aime_2024_0", "question": "What is 1+2?", "ground_truth": "3", "task_type": "aime"}
        sample = SimpleNamespace(text=r"\boxed{3}", token_ids=[1, 2], finish_reason="stop")
        engine = Mock(generate=Mock(return_value=[SimpleNamespace(outputs=[sample])]))
        fake_vllm = SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        cases = [
            (native, ["--model", "fake"], {"load_benchmark": [item]}),
            (adaptive, ["--method", "fixed_merge"], {
                "load_eval_items": [adaptive.EvalItem(0, dataset, split, 0, item)],
                "prepare_merged_models": ({0.7: Path("fake-merged")}, 0.0),
                "load_vllm": engine, "shutdown_vllm": None,
            }),
        ]
        for module, arguments, mocked_returns in cases:
            with self.subTest(evaluator=module.__name__), tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {"vllm": fake_vllm}))
                stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(pad_token="pad")))
                stack.enter_context(patch.object(module, "set_seed"))
                for name, value in mocked_returns.items():
                    stack.enter_context(patch.object(module, name, return_value=value))
                argv = [module.__name__, "--datasets", dataset, "--splits", split,
                        "--prompt-mode", "shared", "--output-dir", td, *arguments]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                module.main()
                summary = json.loads((Path(td) / "summary.json").read_text())
                self.assertEqual(summary["scorer_version"], SCORER_VERSION)
                output = Path(summary["datasets"][0]["output"])
                self.assertEqual(output.parent, Path(td))
                record = json.loads(output.read_text())
                self.assertEqual((record["dataset"], record["split"]), (dataset, split))
                self.assertEqual(record["scorer_version"], SCORER_VERSION)
                self.assertTrue(record["is_correct"])


if __name__ == "__main__":
    unittest.main()
