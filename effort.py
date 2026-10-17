from __future__ import annotations

from typing import Any

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"ultra": "max"}
DEFAULT_EFFORT = "medium"


def _normalize_effort_name(value: Any) -> str:
    effort = str(value or "").strip().lower()
    return EFFORT_ALIASES.get(effort, effort)


def _reasoning_config_for_effort(effort: str) -> dict[str, Any]:
    effort = _normalize_effort_name(effort)
    if effort == "none":
        return {"enabled": False}
    if effort not in EFFORT_ORDER:
        effort = DEFAULT_EFFORT
    return {"enabled": True, "effort": effort}


def _fallback_clamp_effort(effort: str, supported) -> str:
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [lvl for lvl in (str(s).strip().lower() for s in supported) if lvl in EFFORT_ORDER]
    if not supported_norm or requested in supported_norm:
        return effort
    if requested not in EFFORT_ORDER:
        return effort
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_ORDER.index(requested)
    below = [level for level in candidates if EFFORT_ORDER.index(level) < requested_idx]
    return max(below, key=EFFORT_ORDER.index) if below else min(candidates, key=EFFORT_ORDER.index)
