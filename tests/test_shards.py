import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from merge_eval_shards import merge_shards as merge_eval
from merge_h0_shards import merge_shards as merge_h0


class ShardTests(unittest.TestCase):
    def h0_shard(self, root, index, *, item_id=None, k=3, scorer="dalsa-v2"):
        path = root / f"h0_{index}.pt"
        torch.save({"metadata": {"source": f"rollouts_{index}.json", "num_rollouts": k,
                                "num_shards": 2, "shard_index": index, "scorer_version": scorer,
                                "accuracy": index / 2, "total": 1},
                    "hidden_dim": 2, "items": [{"id": item_id or f"x{index}", "h0": torch.zeros(2)}]}, path)
        return str(path)

    def eval_shard(self, root, index, *, item_id=None, temperature=0.6, scorer="dalsa-v2"):
        directory = root / f"eval_{index}"
        directory.mkdir()
        row = {"dataset": "math-ai/aime24", "split": "test/part", "id": item_id or f"x{index}",
               "index": index, "generation_index": 0, "is_correct": index == 0,
               "num_tokens": 10, "finish_reason": "stop", "scorer_version": scorer}
        data = directory / "rows.jsonl"
        data.write_text(json.dumps(row) + "\n")
        summary = {"method": "routing", "prompt_mode": "native", "scorer_version": scorer,
                   "short_model": "short", "long_model": "long",
                   "sampling": {"temperature": temperature, "num_shards": 2, "shard_index": index},
                   "shard": {"num_shards": 2, "shard_index": index},
                   "plan": {"routing_threshold": 0.5, "total_items": index + 1},
                   "groups": [{"num_items": index + 1}],
                   "datasets": [{"dataset": row["dataset"], "split": row["split"], "output": str(data)}]}
        (directory / "summary.json").write_text(json.dumps(summary))
        return str(directory)

    def test_h0_merges_preserves_k_and_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = merge_h0([self.h0_shard(root, i) for i in range(2)], str(root / "out.pt"))
            self.assertEqual(len(result["items"]), 2)
            self.assertEqual(result["metadata"]["num_rollouts"], 3)
            self.assertEqual(result["metadata"]["scorer_version"], "dalsa-v2")
            self.assertNotIn("shard_index", result["metadata"])

    def test_h0_rejects_duplicates_and_inconsistent_settings_before_write(self):
        for mismatch in ("duplicate", "K", "scorer", "missing_shard"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = self.h0_shard(root, 0)
                second = self.h0_shard(root, 1, item_id="x0" if mismatch == "duplicate" else None,
                                       k=4 if mismatch == "K" else 3,
                                       scorer="legacy" if mismatch == "scorer" else "dalsa-v2")
                output = root / "out.pt"
                output.write_bytes(b"existing-output")
                with self.assertRaises(ValueError):
                    merge_h0([first] if mismatch == "missing_shard" else [first, second], str(output))
                self.assertEqual(output.read_bytes(), b"existing-output")

    def test_eval_merges_dataset_alias_and_ignores_shard_specific_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "merged"
            result = merge_eval([self.eval_shard(root, i) for i in range(2)], str(output))
            self.assertEqual(result["datasets"][0]["total"], 2)
            self.assertEqual(result["datasets"][0]["accuracy"], 0.5)
            self.assertEqual(Path(result["datasets"][0]["output"]).name, "math-ai_aime24_test_part_routing_native.jsonl")
            self.assertNotIn("shard_index", result["sampling"])

    def test_eval_rejects_duplicate_or_inconsistent_shards_before_write(self):
        for mismatch in ("duplicate", "temperature", "scorer", "missing_shard"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = self.eval_shard(root, 0)
                second = self.eval_shard(root, 1, item_id="x0" if mismatch == "duplicate" else None,
                                         temperature=0.8 if mismatch == "temperature" else 0.6,
                                         scorer="legacy" if mismatch == "scorer" else "dalsa-v2")
                output = root / "merged"
                with self.assertRaises(ValueError):
                    merge_eval([first] if mismatch == "missing_shard" else [first, second], str(output))
                self.assertFalse(output.exists())

    def test_legacy_eval_scorer_metadata_is_allowed_when_consistently_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.eval_shard(root, i) for i in range(2)]
            for path in paths:
                directory = Path(path)
                summary = json.loads((directory / "summary.json").read_text())
                summary.pop("scorer_version")
                summary.pop("shard")
                (directory / "summary.json").write_text(json.dumps(summary))
                row = json.loads((directory / "rows.jsonl").read_text())
                row.pop("scorer_version")
                (directory / "rows.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(merge_eval(paths, str(root / "merged"))["datasets"][0]["total"], 2)

    def test_legacy_metadata_is_allowed_when_consistently_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for i in range(2):
                path = root / f"legacy_{i}.pt"
                torch.save({"metadata": {}, "hidden_dim": 1, "items": [{"id": str(i)}]}, path)
                paths.append(str(path))
            self.assertEqual(len(merge_h0(paths, str(root / "out.pt"))["items"]), 2)


if __name__ == "__main__":
    unittest.main()
