"""Tests. Run: python -m unittest discover -s src"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets any test runner import the modules in src/

import numpy as np
import pandas as pd

import evaluate as E
import main as M
import router as R


def scored(rows, gold):
    """Write predictions and gold labels to a temp folder and run the evaluator on them."""
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        pd.DataFrame(rows).to_csv(path / "pred.csv", index=False)
        pd.DataFrame(gold).to_csv(path / "gold.csv", index=False)
        return E.evaluate(path / "pred.csv", path / "gold.csv", path / "eval.json")


class RoutingRules(unittest.TestCase):
    def test_never_returns_an_empty_route(self):
        self.assertEqual(R.decide([0.10, 0.30, 0.20], [0.5, 0.5, 0.5]), ["CLIP"])

    def test_every_label_over_its_own_threshold(self):
        self.assertEqual(R.decide([0.40, 0.90, 0.35], [0.36, 0.48, 0.32]), ["FRAME", "CLIP", "VIDEO"])

    def test_cost_metric(self):
        y = np.array([[1, 0, 0], [0, 1, 1]])
        self.assertEqual(R.expected_routing_cost(y, y)["regret_pct"], 0.0)                      # perfect routing
        self.assertEqual(R.expected_routing_cost(y, np.ones_like(y))["erc"], 21.0)              # route everything
        self.assertEqual(R.per_question_cost(y[1:], np.array([[0, 1, 0]]))[0], 4.0 + R.DEFAULT_MISS)  # a miss
        with self.assertRaises(ValueError):
            R.check((1.0, 4.0, 16.0), 10.0)  # a miss must cost at least as much as querying everything

    def test_min_cost_rule(self):
        routes = R.min_cost_routes(np.array([[0.99, 0.01, 0.01], [0.6, 0.6, 0.02]]))
        self.assertEqual(routes.tolist(), [[1, 0, 0], [1, 1, 0]])  # confident FRAME; unsure FRAME vs CLIP: both


class TrainedRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(M.DATA / "train.csv")
        cls.bundle = M.fit_bundle(cls.train.question, cls.train[M.COLS].to_numpy())

    def test_clear_cases_route_correctly(self):
        cases = {"What color is the jacket the teacher is wearing?": ["FRAME"],
                 "What does the dog do after the doorbell rings?": ["CLIP"],
                 "What is the documentary mainly about?": ["VIDEO"],
                 "How many times does the player shoot the ball?": ["CLIP", "VIDEO"]}
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(R.decide(R.scores([question], self.bundle)[0]), expected)

    def test_min_cost_policy_runs_through_the_bundle(self):
        bundle = dict(self.bundle, policy="min_cost", route_costs=R.DEFAULT_COSTS, miss_cost=R.DEFAULT_MISS,
                      calibrators=R.fit_calibrators(R.scores(self.train.question.tolist(), self.bundle),
                                                    self.train[M.COLS].to_numpy()))
        result = R.route("What is the documentary mainly about?", bundle)
        self.assertEqual(result["policy"], "min_cost")
        self.assertIn("VIDEO", result["intent"])

    def test_single_query_is_fast(self):
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            R.scores(["What color is the car when it crashes into the wall?"], self.bundle)
            timings.append((time.perf_counter() - start) * 1000)
        self.assertLess(float(np.median(timings)), 10.0)


class LoadingAndEvaluation(unittest.TestCase):
    def test_ids_keep_leading_zeros(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa.jsonl"
            path.write_text(json.dumps({"video_id": "001", "question_id": "001-1", "question": "x?"}) + "\n")
            self.assertEqual(M.load_questions(str(path)).video_id.iloc[0], "001")

    def test_unverified_labels_are_not_scored(self):
        report = scored([{"video_id": "x", "question_id": "1-1", "intent": "FRAME"}],
                        [{"video_id": "x", "question_id": "1-1", "frame": "", "clip": "", "video": "", "verified": 0}])
        self.assertEqual(report["status"], "pending_verified_labels")
        self.assertNotIn("macro_f1", report)

    def test_coverage_and_absent_classes(self):
        report = scored([{"video_id": "a", "question_id": "1", "intent": "FRAME+CLIP"},
                         {"video_id": "b", "question_id": "2", "intent": "FRAME"}],
                        [{"video_id": "a", "question_id": "1", "frame": 1, "clip": 1, "video": 0, "verified": 1},
                         {"video_id": "b", "question_id": "2", "frame": 0, "clip": 1, "video": 0, "verified": 1}])
        self.assertEqual(report["full_query_coverage"], 0.5)
        perfect = scored([{"video_id": "a", "question_id": "1", "intent": "FRAME"}],
                         [{"video_id": "a", "question_id": "1", "frame": 1, "clip": 0, "video": 0, "verified": 1}])
        self.assertEqual(perfect["macro_f1"], 1.0)  # classes with no labels and no predictions are skipped

    def test_dev_test_split_and_video_level_intervals(self):
        videos = [f"{i:03d}" for i in range(1, 91)]
        dev, test = E.split_videos(videos, "dev"), E.split_videos(videos, "test")
        self.assertEqual((len(dev), dev & test, dev | test), (30, set(), set(videos)))
        rows = [{"video_id": f"{v:03d}", "question_id": f"{v:03d}-{q}", "intent": "FRAME" if q == 1 else "CLIP"}
                for v in range(1, 9) for q in (1, 2)]
        gold = [{"video_id": r["video_id"], "question_id": r["question_id"], "frame": int(r["intent"] == "FRAME"),
                 "clip": int(r["intent"] == "CLIP"), "video": 0, "verified": 1} for r in rows]
        self.assertEqual(scored(rows, gold)["confidence_intervals_95"]["macro_f1"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
