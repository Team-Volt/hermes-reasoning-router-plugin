from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "minilm_router_eval",
        PLUGIN_ROOT / "scripts" / "minilm_router_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n")


class KeywordEmbedder:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "thanks" in lowered or "okay" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "patch" in lowered or "test" in lowered or "implement" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "restart" in lowered or "gateway" in lowered or "mcp" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.1, 0.1, 0.1])
        return vectors


class FixedEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


def test_load_decisions_skips_bad_rows_and_dedupes_by_message_and_effort(tmp_path):
    module = load_eval_module()
    log_path = tmp_path / "reasoning-router.jsonl"
    write_jsonl(
        log_path,
        [
            {"message_preview": "Thanks!", "effort": "low", "reason": "quick", "timestamp": "1"},
            {"message_preview": " thanks! ", "effort": "low", "reason": "duplicate", "timestamp": "2"},
            {"message_preview": "Patch the plugin and run tests", "effort": "high", "timestamp": "3"},
            {"message_preview": "Thanks!", "effort": "medium", "timestamp": "4"},
            {"message_preview": "Unknown effort", "effort": "huge"},
            {"message_preview": "", "effort": "low"},
            "{not json",
        ],
    )

    decisions, stats = module.load_decisions(log_path)

    assert [(d.message, d.effort) for d in decisions] == [
        ("Thanks!", "low"),
        ("Patch the plugin and run tests", "high"),
        ("Thanks!", "medium"),
    ]
    assert stats["total_rows"] == 7
    assert stats["duplicate_rows"] == 1
    assert stats["malformed_rows"] == 1
    assert stats["skipped_rows"] == 2


def test_leave_one_out_eval_uses_injected_embedder_and_reports_confusion():
    module = load_eval_module()
    decisions = [
        module.Decision("Thanks", "low", reason="quick"),
        module.Decision("Okay thanks", "low", reason="quick"),
        module.Decision("Patch the plugin", "high", reason="implementation"),
        module.Decision("Implement tests", "high", reason="implementation"),
        module.Decision("Restart the gateway", "xhigh", reason="service control"),
        module.Decision("Shut down the mcp", "xhigh", reason="service control"),
    ]

    report = module.evaluate_leave_one_out(decisions, embedder=KeywordEmbedder(), top_k=1)

    assert report["total"] == 6
    assert report["evaluated"] == 6
    assert report["correct"] == 6
    assert report["accuracy"] == 1.0
    assert report["confusion"]["low"]["low"] == 2
    assert report["confusion"]["high"]["high"] == 2
    assert report["confusion"]["xhigh"]["xhigh"] == 2


def test_conflicted_examples_surface_near_neighbor_risk_cases():
    module = load_eval_module()
    decisions = [
        module.Decision("Do you need me to restart the gateway?", "medium", reason="question"),
        module.Decision("Please restart the gateway", "xhigh", reason="service control"),
        module.Decision("Could this require changing config?", "medium", reason="question"),
        module.Decision("Patch the plugin config", "high", reason="implementation"),
    ]
    embedder = FixedEmbedder(
        {
            "Do you need me to restart the gateway?": [1.0, 0.0],
            "Please restart the gateway": [0.99, 0.01],
            "Could this require changing config?": [0.0, 1.0],
            "Patch the plugin config": [0.01, 0.99],
        }
    )

    report = module.evaluate_leave_one_out(decisions, embedder=embedder, top_k=1, conflict_limit=10)

    assert report["confusion"]["medium"]["xhigh"] == 1
    assert any(
        item["actual"] == "medium"
        and item["predicted"] == "xhigh"
        and "restart the gateway" in item["message"]
        for item in report["conflicted_examples"]
    )


def test_text_report_is_human_scannable():
    module = load_eval_module()
    report = {
        "total": 2,
        "evaluated": 2,
        "correct": 1,
        "accuracy": 0.5,
        "skipped": 0,
        "load_stats": {"malformed_rows": 0, "skipped_rows": 0, "duplicate_rows": 0},
        "confusion": {"low": {"low": 1, "high": 0}, "high": {"low": 1, "high": 0}},
        "conflicted_examples": [
            {
                "message": "Please restart the gateway",
                "actual": "xhigh",
                "predicted": "medium",
                "confidence": 0.74,
                "nearest": [{"message": "Do you need me to restart?", "effort": "medium", "similarity": 0.99}],
            }
        ],
    }

    text = module.render_text_report(report)

    assert "MiniLM router eval" in text
    assert "accuracy: 50.0%" in text
    assert "conflicted examples" in text
    assert "Please restart the gateway" in text


def test_negative_similarity_neighbors_are_skipped_instead_of_confidently_predicted():
    module = load_eval_module()
    decisions = [
        module.Decision("alpha", "low"),
        module.Decision("beta", "high"),
    ]
    embedder = FixedEmbedder({"alpha": [1.0, 0.0], "beta": [-1.0, 0.0]})

    report = module.evaluate_leave_one_out(decisions, embedder=embedder, top_k=1)

    assert report["evaluated"] == 0
    assert report["skipped"] == 2
    assert report["accuracy"] == 0.0


def test_cli_lexical_json_smoke(tmp_path, capsys):
    module = load_eval_module()
    log_path = tmp_path / "reasoning-router.jsonl"
    write_jsonl(
        log_path,
        [
            {"message_preview": "Thanks", "effort": "low"},
            {"message_preview": "Okay thanks", "effort": "low"},
            {"message_preview": "Patch the plugin", "effort": "high"},
            {"message_preview": "Implement the tests", "effort": "high"},
        ],
    )

    exit_code = module.main([
        "--input",
        str(log_path),
        "--embedder",
        "lexical",
        "--output",
        "json",
        "--top-k",
        "1",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["total"] == 4
    assert output["evaluated"] == 4
    assert "confusion" in output