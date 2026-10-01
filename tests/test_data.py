import unittest
from unittest.mock import patch

from dalsa.data import load_benchmark


class AimeDatasetTests(unittest.TestCase):
    def test_aime24_uses_math_ai_source(self):
        calls = []

        def fake_load_dataset(name, *args, **kwargs):
            calls.append((name, kwargs.get("split")))
            return [{"problem": "Problem 1", "solution": r"\boxed{42}", "id": "0"}]

        with patch("dalsa.data.load_dataset", side_effect=fake_load_dataset):
            rows = load_benchmark("aime24", "test")

        self.assertEqual(calls, [("math-ai/aime24", "test")])
        self.assertEqual(rows[0]["id"], "aime_2024_0")
        self.assertEqual(rows[0]["question"], "Problem 1")
        self.assertEqual(rows[0]["ground_truth"], "42")
        self.assertEqual(rows[0]["source_dataset"], "math-ai/aime24")

    def test_aime25_and_aime26_aliases_use_math_ai_sources(self):
        calls = []

        def fake_load_dataset(name, *args, **kwargs):
            calls.append((name, kwargs.get("split")))
            return [{"Problem": "Problem", "Answer": "7", "ID": "x"}]

        with patch("dalsa.data.load_dataset", side_effect=fake_load_dataset):
            rows25 = load_benchmark("aime_2025", "test")
            rows26 = load_benchmark("math-ai/aime26", "test")

        self.assertEqual(calls, [("math-ai/aime25", "test"), ("math-ai/aime26", "test")])
        self.assertEqual(rows25[0]["id"], "aime_2025_x")
        self.assertEqual(rows26[0]["id"], "aime_2026_x")

    def test_default_aime_test_is_math_ai_aime24(self):
        calls = []

        def fake_load_dataset(name, *args, **kwargs):
            calls.append((name, kwargs.get("split")))
            return [{"problem": "Default AIME", "answer": "123", "id": 5}]

        with patch("dalsa.data.load_dataset", side_effect=fake_load_dataset):
            rows = load_benchmark("aime", "test")

        self.assertEqual(calls, [("math-ai/aime24", "test")])
        self.assertEqual(rows[0]["year"], 2024)


if __name__ == "__main__":
    unittest.main()
