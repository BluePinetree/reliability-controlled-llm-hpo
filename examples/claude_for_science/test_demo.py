"""Verify rejection boundaries and offline evidence semantics."""

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_demo import HERE, ROOT, build_report, validate_candidate, validate_config


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.candidate = {"params": {"learning_rate": 0.0001, "batch_size": 32, "optimizer": "Adam"},
                          "mu": 0.5, "sigma": 0.1, "reason": "fixture"}

    def test_candidate_rejections(self):
        self.assertTrue(validate_candidate(self.candidate, self.config["search_space"])[0])
        for field, value in (("learning_rate", 1.0), ("learning_rate", float("nan")),
                             ("learning_rate", float("inf")), ("batch_size", True),
                             ("batch_size", 16.5), ("batch_size", "32"), ("optimizer", "unknown")):
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(self.candidate)
                candidate["params"][field] = value
                self.assertFalse(validate_candidate(candidate, self.config["search_space"])[0])
        for field, value in (("mu", True), ("mu", float("nan")), ("mu", 1.1),
                             ("sigma", 0), ("sigma", -1), ("sigma", float("inf")), ("reason", [])):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.candidate)
                candidate[field] = value
                self.assertFalse(validate_candidate(candidate, self.config["search_space"])[0])
        for candidate in (None, {}, {**self.candidate, "extra": 1},
                          {**self.candidate, "params": {}},
                          {**self.candidate, "params": {**self.candidate["params"], "extra": 1}}):
            self.assertFalse(validate_candidate(candidate, self.config["search_space"])[0])

    def test_config_rejections(self):
        for key, value in (("mode", "anthropic"), ("candidate_count", 100),
                           ("candidate_count", True), ("seed", -1), ("schema_version", "unknown")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**self.config, key: value})
        for bounds in ([0, 0.001], [0.001, 0.0001], [float("nan"), 0.001]):
            config = copy.deepcopy(self.config)
            config["search_space"]["learning_rate"]["bounds"] = bounds
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_deduplication(self):
        with patch("run_demo.sample_candidate", return_value=self.candidate["params"]):
            rows = build_report(self.config)["validation"]
        self.assertEqual([r["validation_reason"] for r in rows], ["pass", "duplicate", "duplicate", "out_of_bounds"])

    def test_no_network_no_keys_and_evidence_boundary(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
            report = build_report(self.config)
        self.assertEqual(report, build_report(self.config))
        self.assertEqual([r["accepted"] for r in report["validation"]], [True, True, True, False])
        self.assertEqual(report["calibration_gate"], {"passed": False, "reason": "insufficient_records", "record_count": 0})
        self.assertEqual(report["claude_api_calls"], 0)
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["risk_scoring_executed"])
        self.assertIsNone(report["measured_metric"])
        self.assertIsNone(report["selection"]["candidate_id"])
        self.assertIsNone(report["validation"][-1]["candidate"])
        serialized = json.dumps(report)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("reason\": \"fixture", serialized)

    def test_fresh_process_without_provider_imports(self):
        code = '''import sys, runpy, socket
class BlockProviders:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'anthropic', 'dotenv', 'torch', 'transformers'}:
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockProviders())
def deny(*args, **kwargs):
    raise AssertionError('Network forbidden')
socket.socket.connect = deny
socket.create_connection = deny
sys.argv = ['run_demo.py', '--check']
runpy.run_path('examples/claude_for_science/run_demo.py', run_name='__main__')
'''
        env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}
        for seed in ("1", "987"):
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
