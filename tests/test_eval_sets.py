"""Keep the published eval numbers reproducible (see README, Routing accuracy)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate():
    spec = importlib.util.spec_from_file_location("rr_eval", ROOT / "eval" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate()


def test_eval_sets_meet_published_accuracy(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    results = _evaluate()
    assert results["adversarial"]["total"] == 118
    assert results["adversarial"]["passed"] == 118, results["adversarial"]["misses"]
    assert results["tuning"]["total"] == 160 and results["tuning"]["passed"] >= 154, results["tuning"]["misses"]
    assert results["holdout"]["total"] == 150 and results["holdout"]["passed"] >= 143, results["holdout"]["misses"]
