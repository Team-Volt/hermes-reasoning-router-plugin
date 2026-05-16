#!/usr/bin/env python3
"""Offline MiniLM-style evaluator for reasoning-router decision logs.

This is deliberately a *shadow* analysis tool. It reads historical
``reasoning-router.jsonl`` rows, embeds message previews, and runs a
leave-one-out nearest-neighbor evaluation. It does not import the live plugin,
write config, restart services, or affect routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Decision:
    """A normalized routing decision from the JSONL audit log."""

    def __init__(
        self,
        message: str,
        effort: str,
        *,
        reason: str = "",
        timestamp: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.message = str(message or "").strip()
        self.effort = str(effort or "").strip().lower()
        self.reason = str(reason or "").strip()
        self.timestamp = str(timestamp or "").strip()
        self.metadata = dict(metadata or {})

    def preview(self, limit: int = 120) -> str:
        text = " ".join(self.message.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "effort": self.effort,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


Vector = Sequence[float]
Embedder = Callable[[list[str]], Sequence[Vector]]


def default_log_path() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return hermes_home / "logs" / "reasoning-router.jsonl"


def normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def load_decisions(path: str | Path, *, limit: int | None = None) -> tuple[list[Decision], dict[str, int]]:
    """Load and normalize reasoning-router JSONL decisions.

    Deduplication key is normalized message + effort. The same message with a
    different effort is preserved because disagreement is useful for analysis.
    """

    stats = {
        "total_rows": 0,
        "loaded_rows": 0,
        "malformed_rows": 0,
        "skipped_rows": 0,
        "duplicate_rows": 0,
    }
    decisions: list[Decision] = []
    seen: set[tuple[str, str]] = set()
    path = Path(path).expanduser()
    if not path.exists():
        return decisions, stats

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if limit is not None and stats["loaded_rows"] >= limit:
                break
            line = line.strip()
            if not line:
                continue
            stats["total_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                stats["skipped_rows"] += 1
                continue

            message = str(row.get("message_preview") or row.get("message") or "").strip()
            effort = str(row.get("effort") or "").strip().lower()
            if not message or effort not in EFFORTS:
                stats["skipped_rows"] += 1
                continue

            key = (normalize_message(message), effort)
            if key in seen:
                stats["duplicate_rows"] += 1
                continue
            seen.add(key)

            decisions.append(
                Decision(
                    message,
                    effort,
                    reason=str(row.get("reason") or ""),
                    timestamp=str(row.get("timestamp") or ""),
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k not in {"message_preview", "message", "effort", "reason", "timestamp"}
                    },
                )
            )
            stats["loaded_rows"] += 1

    return decisions, stats


def cosine_similarity(left: Vector, right: Vector) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def coerce_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    vectors: list[list[float]] = []
    for vector in value:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vectors.append([float(item) for item in vector])
    return vectors


def embed_messages(decisions: Sequence[Decision], embedder: Embedder | Any) -> list[list[float]]:
    texts = [decision.message for decision in decisions]
    if hasattr(embedder, "encode"):
        raw = embedder.encode(texts)
    else:
        raw = embedder(texts)
    vectors = coerce_vectors(raw)
    if len(vectors) != len(texts):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(texts)} messages")
    return vectors


def nearest_neighbors(
    index: int,
    decisions: Sequence[Decision],
    embeddings: Sequence[Vector],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for other_index, other in enumerate(decisions):
        if other_index == index:
            continue
        similarity = cosine_similarity(embeddings[index], embeddings[other_index])
        neighbors.append(
            {
                "index": other_index,
                "message": other.message,
                "effort": other.effort,
                "similarity": similarity,
            }
        )
    neighbors.sort(key=lambda item: item["similarity"], reverse=True)
    return neighbors[: max(1, int(top_k))]


def predict_effort(neighbors: Sequence[dict[str, Any]]) -> tuple[str | None, float, dict[str, float]]:
    if not neighbors:
        return None, 0.0, {}

    weights: Counter[str] = Counter()
    for neighbor in neighbors:
        # Negative similarities are anti-signal, not weak positive evidence.
        # If every neighbor is <= 0, skip prediction rather than inventing
        # confidence from unrelated rows.
        weight = max(float(neighbor.get("similarity") or 0.0), 0.0)
        if weight <= 0.0:
            continue
        weights[str(neighbor.get("effort"))] += weight

    total = float(sum(weights.values()))
    if total <= 0.0:
        return None, 0.0, {}

    predicted, score = max(weights.items(), key=lambda item: (item[1], -EFFORTS.index(item[0]) if item[0] in EFFORTS else 0))
    confidence = float(score) / total
    return predicted, confidence, {effort: float(score) for effort, score in weights.items()}


def empty_confusion(efforts: Iterable[str] = EFFORTS) -> dict[str, dict[str, int]]:
    labels = list(efforts)
    return {actual: {predicted: 0 for predicted in labels} for actual in labels}


def evaluate_leave_one_out(
    decisions: Sequence[Decision],
    *,
    embedder: Embedder | Any,
    top_k: int = 3,
    conflict_limit: int = 10,
    load_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate nearest-neighbor routing with each decision held out once."""

    decisions = list(decisions)
    confusion = empty_confusion()
    report: dict[str, Any] = {
        "total": len(decisions),
        "evaluated": 0,
        "correct": 0,
        "accuracy": 0.0,
        "skipped": 0,
        "confusion": confusion,
        "conflicted_examples": [],
        "load_stats": dict(load_stats or {}),
    }
    if len(decisions) < 2:
        report["skipped"] = len(decisions)
        return report

    embeddings = embed_messages(decisions, embedder)
    conflicted: list[dict[str, Any]] = []

    for index, decision in enumerate(decisions):
        neighbors = nearest_neighbors(index, decisions, embeddings, top_k=top_k)
        predicted, confidence, votes = predict_effort(neighbors)
        if predicted is None:
            report["skipped"] += 1
            continue

        report["evaluated"] += 1
        confusion.setdefault(decision.effort, {effort: 0 for effort in EFFORTS})
        confusion[decision.effort].setdefault(predicted, 0)
        confusion[decision.effort][predicted] += 1
        is_correct = predicted == decision.effort
        if is_correct:
            report["correct"] += 1

        neighbor_efforts = {str(item["effort"]) for item in neighbors}
        if not is_correct or confidence < 0.67 or len(neighbor_efforts) > 1:
            conflicted.append(
                {
                    "message": decision.message,
                    "actual": decision.effort,
                    "predicted": predicted,
                    "confidence": round(confidence, 4),
                    "votes": {k: round(v, 4) for k, v in votes.items()},
                    "nearest": [
                        {
                            "message": str(item["message"]),
                            "effort": str(item["effort"]),
                            "similarity": round(float(item["similarity"]), 4),
                        }
                        for item in neighbors
                    ],
                }
            )

    evaluated = int(report["evaluated"])
    report["accuracy"] = (float(report["correct"]) / evaluated) if evaluated else 0.0
    conflicted.sort(key=lambda item: (item["actual"] == item["predicted"], item["confidence"]))
    report["conflicted_examples"] = conflicted[: max(0, int(conflict_limit))]
    return report


class SentenceTransformerEmbedder:
    """Runtime MiniLM embedder; imported lazily so tests need no model package."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on local optional package state
            raise RuntimeError(
                "MiniLM embedder requires sentence-transformers. "
                "Install it or rerun with --embedder lexical for a dependency-free smoke."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> Any:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class LexicalHashEmbedder:
    """Tiny dependency-free smoke-test embedder, not a MiniLM substitute."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = max(8, int(dimensions))

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            vectors.append(vector)
        return vectors


def compact_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        actual: {predicted: count for predicted, count in row.items() if count}
        for actual, row in confusion.items()
        if any(row.values())
    }


def render_text_report(report: dict[str, Any]) -> str:
    accuracy = float(report.get("accuracy") or 0.0) * 100.0
    lines = [
        "MiniLM router eval",
        f"total: {report.get('total', 0)}",
        f"evaluated: {report.get('evaluated', 0)}",
        f"correct: {report.get('correct', 0)}",
        f"accuracy: {accuracy:.1f}%",
        f"skipped: {report.get('skipped', 0)}",
    ]
    stats = report.get("load_stats") or {}
    if stats:
        lines.append(
            "load: "
            + ", ".join(
                f"{key}={stats.get(key, 0)}"
                for key in ["total_rows", "loaded_rows", "duplicate_rows", "skipped_rows", "malformed_rows"]
                if key in stats
            )
        )

    lines.append("")
    lines.append("confusion:")
    compact = compact_confusion(report.get("confusion") or {})
    if compact:
        for actual, row in compact.items():
            pieces = ", ".join(f"{predicted}:{count}" for predicted, count in row.items())
            lines.append(f"  {actual} -> {pieces}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("conflicted examples:")
    examples = report.get("conflicted_examples") or []
    if not examples:
        lines.append("  (none)")
    for item in examples:
        lines.append(
            f"  - {item.get('actual')} -> {item.get('predicted')} "
            f"conf={float(item.get('confidence') or 0.0):.2f}: {item.get('message')}"
        )
        nearest = item.get("nearest") or []
        for neighbor in nearest[:3]:
            lines.append(
                f"      neighbor {neighbor.get('effort')} sim={float(neighbor.get('similarity') or 0.0):.2f}: "
                f"{neighbor.get('message')}"
            )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline MiniLM nearest-neighbor evaluator for reasoning-router logs")
    parser.add_argument("--input", default=str(default_log_path()), help="reasoning-router JSONL path")
    parser.add_argument("--top-k", type=int, default=3, help="neighbors to vote over")
    parser.add_argument("--conflict-limit", type=int, default=12, help="max conflicted examples to print")
    parser.add_argument("--limit", type=int, default=None, help="max loaded decisions after dedupe")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    parser.add_argument("--embedder", choices=("minilm", "lexical"), default="minilm")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model for --embedder minilm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decisions, stats = load_decisions(args.input, limit=args.limit)
    try:
        if args.embedder == "lexical":
            embedder: Any = LexicalHashEmbedder()
        else:
            embedder = SentenceTransformerEmbedder(args.model)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    report = evaluate_leave_one_out(
        decisions,
        embedder=embedder,
        top_k=args.top_k,
        conflict_limit=args.conflict_limit,
        load_stats=stats,
    )
    if args.output == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually/CLI smoke
    raise SystemExit(main())
