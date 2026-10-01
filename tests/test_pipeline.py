import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_rerun_uses_only_current_shards_and_sampling_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "run"
            # Files from an earlier four-GPU run must not enter a one-GPU rerun.
            (out / "probe").mkdir(parents=True)
            (out / "probe" / "h0_3.pt").write_text("stale")
            (out / "routing_all_short" / "shards" / "3").mkdir(parents=True)
            log = tmp / "calls"
            log.mkdir()
            stub = tmp / "python_stub"
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys, uuid\n"
                "from pathlib import Path\n"
                "Path(os.environ['CALLS'], str(uuid.uuid4())+'.json').write_text(json.dumps(sys.argv[1:]))\n"
            )
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PYTHON": str(stub), "CALLS": str(log), "OUT": str(out),
                "PAIR": "qwen1.5b", "GPUS": "0", "STAGES": "probe routing native",
                "ROLLOUT_TEMPERATURE": "0.6", "NATIVE_LONG_MAX_MODEL_LEN": "65536",
            }
            subprocess.run(["bash", str(ROOT / "scripts/run_pipeline.sh")], env=env,
                           check=True, capture_output=True, text=True)
            calls = [json.loads(p.read_text()) for p in log.iterdir()]
            h0_merge = next(c for c in calls if c[0] == "scripts/merge_h0_shards.py")
            self.assertEqual(h0_merge[2:h0_merge.index("--output")], [str(out / "probe/h0_0.pt")])
            eval_merges = [c for c in calls if c[0] == "scripts/merge_eval_shards.py"]
            self.assertEqual(len(eval_merges), 2)
            for c in eval_merges:
                output = c[c.index("--output-dir") + 1]
                self.assertEqual(c[2:c.index("--output-dir")], [str(Path(output) / "shards/0")])
            rollout = next(c for c in calls if c[0] == "scripts/generate_rollouts.py")
            self.assertEqual(rollout[rollout.index("--temperature") + 1], "0.6")
            native_long = next(c for c in calls if c[0] == "scripts/evaluate_vllm.py" and c[c.index("--output-dir") + 1].endswith("native_long"))
            self.assertEqual(native_long[native_long.index("--max-model-len") + 1], "65536")


if __name__ == "__main__":
    unittest.main()
