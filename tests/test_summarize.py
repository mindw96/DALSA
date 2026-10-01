import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize as summary
from summarize import summarize  # noqa: E402


def write(run_dir: Path, records: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "out.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def rec(ds, i, ok, tokens, score=None, **extra):
    r = {"dataset": ds, "id": f"{ds}_{i}", "is_correct": ok, "num_tokens": tokens, "finish_reason": "stop"}
    if score is not None:
        r["routing_score"] = score
    r.update(extra)
    return r


class SummarizeTests(unittest.TestCase):
    def test_families_routing_and_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sets = ["gsm8k", "math500", "amc23", "aime24", "aime25", "aime26"]
            # short is right on the easy item, long on the hard item of every benchmark
            write(out / "routing_all_short", [rec(d, i, i == 0, 100, score=0.2 if i == 0 else 0.9) for d in sets for i in (0, 1)])
            write(out / "routing_all_long", [rec(d, i, i == 1, 1000, score=0.2 if i == 0 else 0.9) for d in sets for i in (0, 1)])
            write(out / "dalsa", [rec("gsm8k", 0, True, 50, lambda_scale=0.1, lambda_scale_exact=0.12),
                                  rec("gsm8k", 1, False, 70, lambda_scale=0.7, lambda_scale_exact=0.68)])
            result = summarize(out)

            short = result["routing_all_short"]["families"]
            self.assertEqual(set(short), {"gsm8k", "math500", "amc23", "aime24_26", "avg"})
            self.assertAlmostEqual(short["avg"]["acc"], 50.0)
            self.assertAlmostEqual(short["avg"]["tokens"], 100.0)
            # tau = 0.5 sends only the hard item to the long model -> both correct
            curve = result["routing_threshold_curve"]
            self.assertAlmostEqual(curve["tau=0.5"]["avg"]["acc"], 100.0)
            self.assertAlmostEqual(curve["tau=0.5"]["avg"]["tokens"], 550.0)
            self.assertAlmostEqual(curve["tau=0.1"]["avg"]["acc"], 50.0)
            buckets = result["dalsa"]["buckets"]["gsm8k"]
            self.assertEqual(buckets["buckets"], {"0.1": 1, "0.7": 1})
            self.assertAlmostEqual(buckets["mean_exact_lambda"], 0.40)


class SummaryTests(unittest.TestCase):
    def test_partial_and_complete_aime_and_empty_runs(self):
        stat = lambda acc: {"acc": acc, "tokens": 10, "length_stop": 0, "n": 1}
        partial = {"gsm8k": stat(100), "math500": stat(100), "aime24": stat(0)}
        self.assertAlmostEqual(summary.families(partial)["avg"]["acc"], 200 / 3)
        self.assertEqual(summary.families({"aime24": stat(0)})["aime24"]["acc"], 0)
        self.assertEqual(summary.families({}), {})
        complete = summary.families({**partial, "aime25": stat(30), "aime26": stat(60)})
        self.assertNotIn("aime24", complete)
        self.assertEqual(complete["aime24_26"]["acc"], 30)

    def test_manifest_excludes_stale_files_and_retains_dataset_stats(self):
        row = {"dataset": "aime24", "split": "test", "id": "1", "is_correct": True,
               "num_tokens": 10, "scorer_version": "dalsa-v2"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "current.jsonl").write_text(json.dumps(row) + "\n")
            (run / "stale.jsonl").write_text(json.dumps({**row, "id": "old", "scorer_version": "old"}) + "\n")
            (run / "summary.json").write_text(json.dumps({"datasets": [{"output": "/old/path/current.jsonl"}]}))
            result = summary.summarize(root)["run"]
            self.assertEqual(result["datasets"]["aime24"]["n"], 1)
            self.assertEqual(result["scorer_version"], "dalsa-v2")
            (run / "current.jsonl").write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                summary.load_run(run)
            (run / "summary.json").unlink()
            (run / "current.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "scorer versions"):
                summary.summarize(root)

    def test_routing_requires_same_samples_and_controller_scores(self):
        row = {"dataset": "aime24", "is_correct": False, "num_tokens": 10, "routing_score": 0.4}
        short, long = {"1": row}, {"1": {**row, "is_correct": True}}
        curve = summary.routing_curve(short, long)
        self.assertEqual(curve["tau=0.4"]["avg"]["acc"], 100)
        self.assertEqual(curve["tau=0.5"]["avg"]["acc"], 0)
        with self.assertRaisesRegex(ValueError, "different samples"):
            summary.routing_curve(short, {})
        with self.assertRaisesRegex(ValueError, "inconsistent routing score"):
            summary.routing_curve(short, {"1": {**row, "routing_score": 0.5}})


if __name__ == "__main__":
    unittest.main()
