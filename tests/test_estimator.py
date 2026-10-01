import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dalsa import estimator


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EstimatorTests(unittest.TestCase):
    def test_extract_preserves_actual_k_and_rollout_provenance(self):
        script = load_script("extract_h0")
        metadata = {"num_rollouts": 3, "dataset": "test", "temperature": 0.7,
                    "scorer_version": "dalsa-v2", "num_shards": 2, "shard_index": 1}
        rows = [{"id": str(i), "prompt": "q", "is_hard": True, "correct_count": i} for i in range(3)]
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=2), eval=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "rollouts.json", Path(tmp) / "h0.pt"
            source.write_text(json.dumps({"metadata": metadata, "data": rows}))
            with patch.object(script, "load_tokenizer", return_value=object()), \
                 patch.object(script, "load_causal_lm", return_value=model), \
                 patch.object(script, "extract_batch", return_value=torch.zeros(3, 2)), \
                 patch.object(sys, "argv", ["extract_h0.py", "--input", str(source), "--model", "mock", "--output", str(output)]):
                script.main()
            data = torch.load(output, map_location="cpu")
        for key, value in metadata.items():
            self.assertEqual(data["metadata"][key], value)
        torch.testing.assert_close(estimator.soft_fail_targets(data), torch.tensor([1.0, 2 / 3, 1 / 3]))
        for item in data["items"]:
            item["correct_count"] = 0
        torch.testing.assert_close(estimator.soft_fail_targets(data), torch.ones(3))

    def test_unknown_invalid_or_conflicting_k_is_rejected(self):
        data = {"metadata": {}, "items": [{"correct_count": 3}]}
        with self.assertRaisesRegex(ValueError, "--num-rollouts"):
            estimator.infer_num_rollouts(data)
        self.assertEqual(estimator.infer_num_rollouts(data, 3), 3)
        for value in (0, -1, 2.5, True, "3", None):
            with self.subTest(k=value), self.assertRaisesRegex(ValueError, "positive integer"):
                estimator.infer_num_rollouts({**data, "metadata": {"num_rollouts": value}})
        for changed in (
            {"metadata": {"num_rollouts": 3, "K": 2}, "items": []},
            {"metadata": {"num_rollouts": 3}, "items": [{"num_rollouts": 2}]},
            {"metadata": {}, "items": [{"num_rollouts": 3}, {"num_rollouts": 2}]},
        ):
            with self.subTest(data=changed), self.assertRaisesRegex(ValueError, "Conflicting"):
                estimator.infer_num_rollouts(changed)

    def test_correct_counts_are_validated_instead_of_clipped(self):
        for value in (-1, 4, 1.5, True, "2", None):
            data = {"metadata": {"num_rollouts": 3}, "items": [{"correct_count": value}]}
            with self.subTest(count=value), self.assertRaisesRegex(ValueError, "correct_count"):
                estimator.soft_fail_targets(data)

    def test_legacy_cli_k_applies_to_train_and_validation_without_overwriting_known_k(self):
        script = load_script("train_estimator")
        data = {"hidden_dim": 2, "items": [{"h0": torch.zeros(2), "is_hard": True, "correct_count": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "h0.pt"
            torch.save(data, source)
            argv = ["train_estimator.py", "--train", str(source), "--val", str(source),
                    "--output", str(Path(tmp) / "probe"), "--num-rollouts", "3"]
            with patch.object(sys, "argv", argv), patch.object(script, "set_seed"), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(script, "train_estimator", return_value={}) as train:
                script.main()
                self.assertEqual(train.call_args.kwargs["train_data"]["metadata"]["num_rollouts"], 3)
                self.assertEqual(train.call_args.kwargs["val_data"]["metadata"]["num_rollouts"], 3)
                data["metadata"] = {"num_rollouts": 4}
                torch.save(data, source)
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    script.main()
                self.assertEqual(train.call_count, 1)

    def test_mlp_validation_matches_loaded_checkpoint_and_disables_dropout(self):
        torch.manual_seed(42)
        data = {"hidden_dim": 4, "metadata": {"num_rollouts": 3}, "items": [
            {"h0": torch.randn(4), "is_hard": bool(i % 2), "correct_count": 0 if i % 2 else 3}
            for i in range(16)
        ]}
        model = estimator.ValueNetwork(4)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(estimator, "build_estimator_model", return_value=model):
                metrics = estimator.train_estimator(data, data, tmp, epochs=1, batch_size=16, device="cpu")
            self.assertFalse(model.training)
            loaded = estimator.DifficultyEstimator.load(tmp, device="cpu")
            self.assertEqual(metrics, estimator.evaluate_estimator(loaded.model, data, metrics["threshold"], "cpu"))
        model.train()
        first = estimator.evaluate_estimator(model, data, 0.5, "cpu")
        self.assertFalse(model.training)
        model.train()
        self.assertEqual(first, estimator.evaluate_estimator(model, data, 0.5, "cpu"))
        model.train()
        threshold = estimator.find_best_threshold(model, data, "cpu")
        self.assertFalse(model.training)
        self.assertEqual(threshold, estimator.find_best_threshold(model, data, "cpu"))


if __name__ == "__main__":
    unittest.main()
