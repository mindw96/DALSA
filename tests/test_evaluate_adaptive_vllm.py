import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_adaptive_vllm.py"
SPEC = importlib.util.spec_from_file_location("evaluate_adaptive_vllm", SCRIPT)
evaluate_adaptive = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluate_adaptive
SPEC.loader.exec_module(evaluate_adaptive)


class LambdaScoreRemapTests(unittest.TestCase):
    def test_quantile_lambda_score_remap_clips_to_unit_interval(self):
        remap = {"mode": "quantile", "score_lo": 0.2, "score_hi": 0.6}

        self.assertEqual(evaluate_adaptive.apply_lambda_score_remap(0.1, remap), 0.0)
        self.assertAlmostEqual(evaluate_adaptive.apply_lambda_score_remap(0.4, remap), 0.5)
        self.assertEqual(evaluate_adaptive.apply_lambda_score_remap(0.9, remap), 1.0)

    def test_assign_groups_uses_remapped_score_for_bucket_lambda(self):
        args = Namespace(
            method="dalsa_bucketed",
            lambda_values="0.1,0.3,0.5",
            min_lambda=0.1,
            max_lambda=0.5,
            lambda_buckets=3,
            merge_base="short",
        )
        rows = [
            evaluate_adaptive.EvalItem(
                global_index=0,
                dataset="gsm8k",
                split="test",
                dataset_index=0,
                item={"id": "x", "question": "q", "ground_truth": "1"},
            )
        ]
        groups, extra, lambdas = evaluate_adaptive.assign_groups(
            args,
            rows,
            p_hard_by_index={0: 0.4},
            routing_threshold=0.5,
            lambda_score_remap={"mode": "quantile", "score_lo": 0.2, "score_hi": 0.6},
        )

        self.assertEqual(lambdas, [0.1, 0.3, 0.5])
        self.assertEqual(list(groups), ["lambda_0.3000"])
        self.assertAlmostEqual(extra[0]["p_hard"], 0.4)
        self.assertAlmostEqual(extra[0]["lambda_score"], 0.5)
        self.assertAlmostEqual(extra[0]["lambda_scale_exact"], 0.3)

    def test_assign_groups_uses_remapped_score_for_routing(self):
        args = Namespace(method="routing")
        rows = [
            evaluate_adaptive.EvalItem(
                global_index=0,
                dataset="gsm8k",
                split="test",
                dataset_index=0,
                item={"id": "x", "question": "q", "ground_truth": "1"},
            ),
            evaluate_adaptive.EvalItem(
                global_index=1,
                dataset="gsm8k",
                split="test",
                dataset_index=1,
                item={"id": "y", "question": "q", "ground_truth": "2"},
            ),
        ]
        groups, extra, lambdas = evaluate_adaptive.assign_groups(
            args,
            rows,
            p_hard_by_index={0: 0.1, 1: 0.4},
            routing_threshold=0.5,
            lambda_score_remap={"mode": "quantile", "score_lo": 0.2, "score_hi": 0.6},
        )

        self.assertEqual(lambdas, [])
        self.assertEqual([row.global_index for row in groups["short"]], [0])
        self.assertEqual([row.global_index for row in groups["long"]], [1])
        self.assertAlmostEqual(extra[0]["routing_score"], 0.0)
        self.assertAlmostEqual(extra[1]["routing_score"], 0.5)
        self.assertEqual(extra[1]["routing_score_remap"], "quantile")


class CalibrationCacheTests(unittest.TestCase):
    def args(self, tmp, scope="acm_targets"):
        return Namespace(short_model="short", long_model="long", seed=42, merge_scope=scope,
                         calibration=str(Path(tmp) / "calibration.pt"), merged_model_cache=tmp,
                         theta=1.0, calibration_dataset="gsm8k", calibration_split="train",
                         calibration_samples=128, mi_clusters=6, merge_base="short",
                         disable_layer_consensus=False, fim_random_samples=32, fim_seq_len=64,
                         fim_batch_size=1, fim_alpha_min=0.2, fim_alpha_max=0.8,
                         fim_temperature=2.0, fim_device="cuda")

    def test_calibration_request_is_saved_and_mismatches_rejected(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock, patch
        for scope, kind in (("acm_targets", "acm"), ("fim_full", "fim")):
            with self.subTest(scope=scope), TemporaryDirectory() as tmp:
                args = self.args(tmp, scope)
                info = Mock(metadata={})
                getter = evaluate_adaptive.get_calibration if kind == "acm" else evaluate_adaptive.get_fim_calibration
                cls = evaluate_adaptive.ACMCalibrationInfo if kind == "acm" else evaluate_adaptive.FIMCalibrationInfo
                with patch.object(evaluate_adaptive, f"precompute_{kind}_calibration", return_value=info) as compute:
                    self.assertIs(getter(args), info)
                    info.save.assert_called_once_with(args.calibration)
                    self.assertEqual(info.metadata["request"], evaluate_adaptive.calibration_request(args))
                    Path(args.calibration).write_text("cache")
                    with patch.object(cls, "load", return_value=info):
                        self.assertIs(getter(args), info)
                        args.seed += 1
                        with self.assertRaisesRegex(ValueError, "new --calibration"):
                            getter(args)
                        args.seed -= 1
                        info.metadata.clear()
                        with self.assertRaisesRegex(ValueError, "provenance"):
                            getter(args)
                    self.assertEqual(compute.call_count, 1)

    def test_merge_cache_changes_with_settings_and_replaced_calibration(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            args = self.args(tmp)
            path = Path(args.calibration)
            path.write_text("first")
            original = evaluate_adaptive.merged_cache_path(args, 0.3)
            args.theta += 0.1
            self.assertNotEqual(original, evaluate_adaptive.merged_cache_path(args, 0.3))
            args.theta -= 0.1
            path.write_text("replacement")
            self.assertNotEqual(original, evaluate_adaptive.merged_cache_path(args, 0.3))
            dataset = Path(tmp) / "calibration.json"
            dataset.write_text("[]")
            args.calibration_dataset = str(dataset)
            before = evaluate_adaptive.calibration_request(args)
            dataset.write_text('[{"question": "changed"}]')
            self.assertNotEqual(before, evaluate_adaptive.calibration_request(args))
            args.merge_scope = "fim_full"
            before = evaluate_adaptive.merged_cache_path(args, 0.3)
            args.seed += 1
            self.assertNotEqual(before, evaluate_adaptive.merged_cache_path(args, 0.3))


if __name__ == "__main__":
    unittest.main()
