"""Automatic reasoning-effort router for Hermes gateway sessions.

This plugin intentionally does not patch Hermes core. It uses the existing
``pre_gateway_dispatch`` hook and the gateway's session-scoped reasoning
override mechanism. The gateway later resolves that override and sets
``agent.reasoning_config`` before the provider request is built, so this changes
the real backend reasoning parameter rather than prompt-injecting advice.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - Hermes normally depends on PyYAML
    yaml = None

logger = logging.getLogger(__name__)

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_CONFIG = {
    "enabled": True,
    "default": "medium",
    "min": "low",
    "max": "xhigh",
    # systemd/journald logging through the normal Hermes gateway logger
    "log_decisions": True,
    # persistent JSONL audit trail for later inspection or external audits
    "decision_log": False,
    "decision_log_path": "logs/reasoning-router.jsonl",
    "low_char_limit": 80,
    "xhigh_high_match_threshold": 4,
    # Carry task complexity across terse approvals like "yes" / "go ahead".
    "pending_intent_enabled": True,
    "pending_intent_ttl_minutes": 30,
}

# Explicit xhigh means: slow down; this is multi-system, risky, architectural,
# security-sensitive, or asks for unusually complete execution.
_XHIGH_PATTERNS = (
    r"\b(xhigh|extra\s*high|maximum\s+reasoning|think\s+hard(?:er)?)\b",
    r"\b(be\s+thorough|flesh\s+out|boil\s+the\s+ocean|do\s+the\s+whole\s+thing|end\s+to\s+end)\b",
    r"\b(architecture|architectural|design\s+decision|tradeoff|strategy|migration\s+plan)\b",
    r"\b(security|auth|oauth|credential|secret|permission|token|ssrf|injection)\b",
    r"\b(production|rollback\s+safety|rollback-safe|data\s+loss|incident|outage)\b",
    r"\b(restart\s+the\s+gateway|gateway\s+restart|restart\s+hermes|systemd\s+restart)\b",
    r"\b(multi[-\s]?system|cross[-\s]?system|multiple\s+systems|orchestrat(?:e|ion))\b",
    r"\b(?:delete|remove|purge)\b.{0,160}\bfork\b.{0,220}\b(?:update|switch|migrate|roll\s*out|rollout)\b.{0,120}\b(?:ha|home\s*assistant|hacs)\b",
    r"\b(?:make\s+pr'?s?|open\s+pr'?s?|merge)\b.{0,160}\bfork\b.{0,220}\b(?:update|switch|migrate)\b.{0,120}\b(?:ha|home\s*assistant|hacs)\b",
    r"\b(?:update|switch|migrate)\b.{0,80}\b(?:ha|home\s*assistant|hacs)\b.{0,160}\b(?:new\s+)?fork\b",
    r"\b(?:copy|sync|migrate|import|backfill|mirror)\b.{0,180}\b(?:gbrain|hindsight)\b.{0,180}\b(?:gbrain|hindsight)\b",
    r"\b(?:gbrain|hindsight)\b.{0,180}\b(?:copy|sync|migrate|import|backfill|mirror)\b.{0,180}\b(?:gbrain|hindsight)\b",
)

_DOCS_POLISH_PATTERNS = (
    r"\b(readme|docs?|documentation|install(?:ation)?\s+instructions?|prompt|copy[-\s]?paste\s+prompt|wording)\b",
    r"\b(overly\s+descriptive|wording|generic|standard[is]ed|style|mention|document|macos|linux|systemctl|restart\s+instructions?|ask\s+the\s+user\s+to\s+restart)\b",
)

_DOCS_POLISH_RISK_PATTERNS = (
    r"\b(security|auth|oauth|credential|secret|permission|token|ssrf|injection)\b",
    r"\b(production|deploy|deployment|rollback\s+safety|rollback-safe|data\s+loss|incident|outage)\b",
    r"\b(multi[-\s]?system|cross[-\s]?system|multiple\s+systems|orchestrat(?:e|ion))\b",
)

_IMPLEMENTATION_APPROVAL_PATTERNS = (
    r"\b(?:go\s+ahead|do\s+it|proceed|ship\s+it|make\s+it\s+so)\b.{0,120}\b(?:apply|patch|change|edit|tweak|fix|implement|configure)\b",
    r"\b(?:go\s+ahead|do\s+it|proceed|ship\s+it|make\s+it\s+so)\b.{0,120}\b(?:fork|start\s+working|work\s+on|update\s+(?:ha|home\s*assistant|hacs)?\s*to\s+use)\b",
    r"\bprevent\s+under[-\s]?routing\b",
)

_HIGH_PATTERN_GROUPS = {
    "implementation": (
        r"\b(implement|build|add|create|modify|change|edit|patch|refactor|flesh\s+out)\b",
    ),
    "setup_config": (
        r"\b(set\s*up|setup|configure|install|enable|disable|automation)\b",
        r"\b(config|configuration|yaml|env|plugin|hook)\b",
    ),
    "state_migration": (
        r"\b(update|upgrade|migrate|migration|schema|database|rollback|backup|restore)\b",
    ),
    "debug_forensics": (
        r"\b(debug|fix|troubleshoot|investigate|root\s*cause|forensic|why\s+did|failure)\b",
    ),
    "diagnostic review": (
        r"\b(?:carefully\s+)?(?:review|inspect)\b.{0,140}\b(?:lcm\s+db|lifecycle\s+fragmentation|lifecycle\s+rows|session\s+lifecycle|context\s+engine)\b",
        r"\b(?:lcm\s+db|lifecycle\s+fragmentation|lifecycle\s+rows|session\s+lifecycle|context\s+engine)\b.{0,140}\b(?:review|inspect|diagnos(?:e|is|tic)|repair|deal\s+with)\b",
    ),
    "hermes_internals": (
        r"\b(gateway|transport|provider|reasoning|request\s*parameter|session\s+override|pre_gateway_dispatch)\b",
    ),
    "ops": (
        r"\b(systemd|service|restart|deploy|auth|oauth|credential|production)\b",
    ),
    "destructive": (
        r"\b(delete|remove|purge|destructive|irreversible|data\s+loss)\b",
    ),
    "verification": (
        r"\b(test|tests|verify|verification|smoke\s*test|lint|compile|restart)\b",
    ),
    "logging_audit": (
        r"\b(log|logs|logging|audit|jsonl|persistent)\b",
    ),
}

_MEDIUM_PATTERNS = (
    r"\b(check|inspect|look\s+at|search|find|compare|summarize|research)\b",
    r"\b(home\s*assistant|dashboard|entity|sensor|switch|notify)\b",
    r"\b(file|config|log|status|process|port)\b",
)

# Short feasibility/design follow-ups in an active technical conversation often
# look deceptively tiny, but answering them correctly requires architectural
# context. Check these before the generic "short message => low" fallback.
_MEDIUM_TECH_FOLLOWUP_PATTERNS = (
    r"\b(?:does|would|will|could|can)\b.{0,80}\b(?:require|need|involve|mean|support)\b.{0,80}\b(?:source|code|config|configuration|plugin|hook|cron|scheduler|gateway|hermes)\b",
    r"\b(?:source|code|config|configuration|plugin|hook|cron|scheduler|gateway|hermes)\b.{0,80}\b(?:require|need|involve|mean|support|possible|clean\s+way)\b",
    r"\b(?:clean\s+way|right\s+way|best\s+way)\b.{0,80}\b(?:source|code|config|configuration|plugin|hook|cron|scheduler|gateway|hermes|automation)\b",
)

_MEDIUM_OPINION_PATTERNS = (
    r"\b(?:honest\s+opinion|your\s+opinion|what\s+do\s+you\s+think|your\s+take|thoughts\s+on|is\s+there\s+(?:actually\s+)?value\s+in)\b",
)

_LOW_PATTERNS = (
    r"^(thanks|thank you|ok|okay|yes|no|yep|nope|cool|nice)[.!?\s]*$",
    r"\b(what\s+time|what\s+date|who\s+is|what\s+is)\b",
    r"\b(quick|brief|one\s+sentence|short answer)\b",
)

_COMPILED_XHIGH = tuple(re.compile(pattern, re.I) for pattern in _XHIGH_PATTERNS)
_COMPILED_DOCS_POLISH = tuple(re.compile(pattern, re.I) for pattern in _DOCS_POLISH_PATTERNS)
_COMPILED_DOCS_POLISH_RISK = tuple(
    re.compile(pattern, re.I) for pattern in _DOCS_POLISH_RISK_PATTERNS
)
_COMPILED_IMPLEMENTATION_APPROVAL = tuple(
    re.compile(pattern, re.I) for pattern in _IMPLEMENTATION_APPROVAL_PATTERNS
)
_COMPILED_HIGH_GROUPS = {
    name: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for name, patterns in _HIGH_PATTERN_GROUPS.items()
}
_COMPILED_MEDIUM = tuple(re.compile(pattern, re.I) for pattern in _MEDIUM_PATTERNS)
_COMPILED_MEDIUM_TECH_FOLLOWUP = tuple(
    re.compile(pattern, re.I) for pattern in _MEDIUM_TECH_FOLLOWUP_PATTERNS
)
_COMPILED_MEDIUM_OPINION = tuple(
    re.compile(pattern, re.I) for pattern in _MEDIUM_OPINION_PATTERNS
)
_COMPILED_LOW = tuple(re.compile(pattern, re.I) for pattern in _LOW_PATTERNS)
_AFFIRMATIVE_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"^(?:y|yes|yep|yeah|ok|okay|sure|approved|affirmative)[.!?\s]*$",
        r"^(?:go ahead|do it|proceed|ship it|sounds good|that works|make it so|let'?s do it)[.!?\s]*$",
        r"^(?:go ahead|do it|proceed|ship it|sounds good|that works|make it so|let'?s do it)\s+(?:and\s+)?(?:do|take|run|apply|execute|follow)?\s*(?:the\s+)?(?:next\s+step|recommendation|recommended\s+plan|plan)[.!?\s]*$",
        r"^(?:yes|yep|yeah|ok|okay|sure)[,\s]+(?:go ahead|do it|please|proceed|ship it|that works).*$",
    )
)
_REJECTION_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"^(?:n|no|nope|nah|cancel|stop|wait|not yet|hold off|skip|nevermind|never mind)[.!?\s]*$",
        r"\b(?:do not|don'?t|cancel|stop|hold off|not yet|never mind|nevermind)\b",
    )
)
_PROCEED_ACTION_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:want me to|would you like me to|should I|shall I|do you want me to)\b.{0,160}\b(?:proceed|build|implement|make|create|set\s*up|configure|install|apply|patch|change|edit|run|test|verify|restart|deploy|migrate|delete|remove|execute|start)\b",
        r"\b(?:say|reply|tell me)\s+(?:yes|go|proceed|approved).{0,160}\b(?:proceed|build|implement|start|apply|make|create|set\s*up|run|deploy)\b",
        r"\b(?:I can|I’ll|I'll|I will|next I can)\b.{0,160}\b(?:build|implement|patch|set\s*up|configure|install|apply|make the changes|run the tests|verify|restart|deploy|migrate|execute)\b.{0,80}\?",
        r"\b(?:next\s+step|recommended\s+next\s+step|my\s+recommendation|recommended\s+sequence)\b.{0,220}\b(?:classif(?:y|ication)|split|repair|patch|implement|apply|run|verify|review|inspect|prune|rename|migrate|delete|create|build|set\s*up|configure|produce)\b",
    )
)
_RUNTIME_CONFIG_OVERRIDE: dict[str, Any] | None = None
_PENDING_INTENTS: dict[str, dict[str, Any]] = {}


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_command(
        "reasoning-router",
        reasoning_router_command,
        description="Toggle/status/configure automatic reasoning effort routing",
        args_hint="status|on|off|min|max|default|threshold|pending|log|recent|test <message>",
    )


def pre_gateway_dispatch(event=None, gateway=None, session_store=None, **_kwargs):
    """Route a gateway message to a reasoning effort.

    Return shape follows Hermes' pre_gateway_dispatch contract. We never skip or
    rewrite user messages; the plugin only mutates the gateway's per-session
    reasoning override before normal dispatch continues.
    """
    if event is None or gateway is None:
        return {"action": "allow"}

    if bool(getattr(event, "internal", False)):
        return {"action": "allow"}

    text = str(getattr(event, "text", "") or "")
    if not text.strip():
        return {"action": "allow"}

    # Built-in/plugin slash commands should keep their own semantics. In
    # particular, /reasoning must be able to set manual state without us racing
    # it from the pre-dispatch hook.
    if text.lstrip().startswith("/"):
        return {"action": "allow"}

    config = _router_config(gateway)
    if not _truthy(config.get("enabled", True)):
        return {"action": "allow"}

    session_key = _session_key_for(event, gateway, session_store)
    if not session_key:
        logger.debug("reasoning-router: no session key; allowing without override")
        return {"action": "allow"}

    effort, reason, pending_intent = _effective_effort_for_message(
        text,
        config,
        session_store=session_store,
        session_key=session_key,
    )
    reasoning_config = _reasoning_config_for_effort(effort)

    _set_reasoning_override(gateway, session_key, reasoning_config)
    decision = _record_decision(
        gateway,
        session_key,
        effort,
        reason,
        text,
        event=event,
        pending_intent=pending_intent,
    )

    if _truthy(config.get("log_decisions", True)):
        logger.info(
            "reasoning-router: session=%s effort=%s reason=%s",
            session_key,
            effort,
            reason,
        )

    if _truthy(config.get("decision_log", False)):
        _append_decision_log(config, decision)

    return {"action": "allow"}


def post_llm_call(
    session_id: str | None = None,
    user_message: str | None = None,
    assistant_response: str | None = None,
    conversation_history=None,
    model: str | None = None,
    platform: str | None = None,
    **_kwargs,
):
    """Arm a one-shot pending intent when the assistant asks to proceed.

    This fixes the common continuation turn: assistant gives a substantial plan
    and asks for approval, user replies "yes", and the next gateway dispatch must
    inherit the planned task's effort instead of classifying literal "yes" as low.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return None

    config = _read_router_config_from_disk()
    if not _truthy(config.get("enabled", True)):
        return None
    if not _truthy(config.get("pending_intent_enabled", True)):
        _PENDING_INTENTS.pop(sid, None)
        return None

    response = str(assistant_response or "")
    if not _assistant_asks_to_proceed(response):
        return None

    original = str(user_message or "")
    effort = _max_effort(
        (
            classify_message(original, config)[0],
            classify_message(response, config)[0],
            "high",
        ),
        config,
    )
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(minutes=_pending_intent_ttl_minutes(config))
    pending = {
        "session_id": sid,
        "effort": effort,
        "reason": "assistant asked for approval to proceed with a substantive task",
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "platform": platform or "",
        "model": model or "",
        "user_preview": _preview(original),
        "assistant_preview": _preview(response),
        "consumed": False,
    }
    _PENDING_INTENTS[sid] = pending
    logger.info(
        "reasoning-router: armed pending intent session_id=%s effort=%s platform=%s",
        sid,
        effort,
        platform or "",
    )
    return None


def reasoning_router_command(raw_args: str = "") -> str:
    """Discord/CLI slash command for the router.

    Registered as `/reasoning-router`. Config changes are written to
    `~/.hermes/reasoning-router/config.yaml` and mirrored into this module's runtime override so
    they affect the next gateway message without waiting for a restart.
    """
    args = (raw_args or "").strip()
    if not args or args.lower() == "status":
        cfg = _read_router_config_from_disk()
        return _format_status(cfg)

    parts = args.split(maxsplit=1)
    command = parts[0].strip().lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if command in {"help", "?"}:
        return (
            "Usage: `/reasoning-router status|on|off|min <effort>|max <effort>|"
            "default <effort>|threshold <N>|pending on|pending off|log on|log off|recent [N]|test <message>`\n"
            "Efforts: none, minimal, low, medium, high, xhigh."
        )

    if command in {"on", "enable", "enabled"}:
        _update_router_config({"enabled": True})
        return "Reasoning router enabled."

    if command in {"off", "disable", "disabled"}:
        _update_router_config({"enabled": False})
        return "Reasoning router disabled. Use `/reasoning-router on` to re-enable."

    if command in {"min", "max", "default"}:
        effort = value.lower()
        if effort not in EFFORT_ORDER:
            return f"Invalid effort `{value}`. Use one of: {', '.join(EFFORT_ORDER)}."
        _update_router_config({command: effort})
        return f"Reasoning router {command} effort set to {effort}."

    if command in {"log", "decision-log", "jsonl"}:
        lowered = value.lower()
        if lowered in {"on", "enable", "enabled", "true", "1", "yes"}:
            _update_router_config({"decision_log": True})
            return f"Reasoning router persistent decision log enabled: {_decision_log_path(_read_router_config_from_disk())}"
        if lowered in {"off", "disable", "disabled", "false", "0", "no"}:
            _update_router_config({"decision_log": False})
            return "Reasoning router persistent decision log disabled."
        return "Usage: `/reasoning-router log on|off`"

    if command in {"threshold", "xhigh-threshold"}:
        threshold = _safe_int(value, 0)
        if threshold < 1:
            return "Usage: `/reasoning-router threshold <N>` where N is at least 1."
        _update_router_config({"xhigh_high_match_threshold": threshold})
        return f"Reasoning router xhigh threshold set to {threshold} high-complexity categories."

    if command in {"pending", "continuation", "latch"}:
        lowered = value.lower()
        if lowered in {"on", "enable", "enabled", "true", "1", "yes"}:
            _update_router_config({"pending_intent_enabled": True})
            return "Reasoning router pending-intent inheritance enabled."
        if lowered in {"off", "disable", "disabled", "false", "0", "no"}:
            _update_router_config({"pending_intent_enabled": False})
            _PENDING_INTENTS.clear()
            return "Reasoning router pending-intent inheritance disabled and cleared."
        return "Usage: `/reasoning-router pending on|off`"

    if command in {"recent", "tail", "decisions"}:
        cfg = _read_router_config_from_disk()
        limit = _safe_int(value, 5) if value else 5
        return _format_recent_decisions(cfg, limit=limit)

    if command == "test":
        if not value:
            return "Usage: `/reasoning-router test <message>`"
        cfg = _read_router_config_from_disk()
        effort, reason = classify_message(value, cfg)
        return f"That message would route to {effort}: {reason}."

    return (
        f"Unknown reasoning-router command `{command}`. "
        "Use `/reasoning-router help`."
    )


def classify_message(text: str, config: dict[str, Any] | None = None) -> tuple[str, str]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()

    # Documentation/install-prompt wording can mention operational words like
    # "restart the gateway" or "systemctl" without asking us to touch live ops.
    # Keep that at medium unless other non-docs risk categories dominate.
    if _is_docs_polish_request(lowered):
        return _clamp_effort("medium", cfg), "documentation wording/install-prompt polish"

    # Strongest wins. Avoid low-routing a short sentence like "go ahead and set
    # up the automation" just because it is brief.
    if _matches(_COMPILED_XHIGH, lowered):
        return _clamp_effort("xhigh", cfg), "matched xhigh complexity/risk keywords"

    if _matches(_COMPILED_IMPLEMENTATION_APPROVAL, lowered):
        return _clamp_effort("high", cfg), "matched implementation approval/tweak request"

    high_groups = _matched_high_groups(lowered)
    threshold = _safe_int(cfg.get("xhigh_high_match_threshold"), DEFAULT_CONFIG["xhigh_high_match_threshold"])
    if len(high_groups) >= threshold:
        return (
            _clamp_effort("xhigh", cfg),
            f"matched multiple high-complexity categories: {', '.join(high_groups)}",
        )

    if high_groups:
        return (
            _clamp_effort("high", cfg),
            f"matched high-complexity category: {', '.join(high_groups)}",
        )

    if _matches(_COMPILED_MEDIUM_TECH_FOLLOWUP, lowered):
        return _clamp_effort("medium", cfg), "matched technical feasibility/design follow-up"

    if _matches(_COMPILED_MEDIUM_OPINION, lowered):
        return _clamp_effort("medium", cfg), "matched opinion/take request"

    if _matches(_COMPILED_LOW, lowered) or len(normalized) <= _safe_int(cfg.get("low_char_limit"), 80):
        return _clamp_effort("low", cfg), "quick/simple message"

    if _matches(_COMPILED_MEDIUM, lowered):
        return _clamp_effort("medium", cfg), "matched normal tool/status keywords"

    default = str(cfg.get("default") or DEFAULT_CONFIG["default"]).lower()
    if default not in EFFORT_ORDER:
        default = DEFAULT_CONFIG["default"]
    return _clamp_effort(default, cfg), "default route"


def _effective_effort_for_message(
    text: str,
    config: dict[str, Any],
    *,
    session_store=None,
    session_key: str = "",
) -> tuple[str, str, dict[str, Any] | None]:
    effort, reason = classify_message(text, config)
    if not _truthy(config.get("pending_intent_enabled", True)):
        return effort, reason, None

    session_id = _session_id_for_key(session_store, session_key)
    pending = _active_pending_intent(session_id, config) if session_id else None
    if not pending:
        return effort, reason, None

    if _is_rejection(text):
        _consume_pending_intent(session_id)
        return effort, f"rejected pending task; {reason}", pending

    if _is_affirmative(text):
        _consume_pending_intent(session_id)
        inherited = str(pending.get("effort") or "high")
        routed = _max_effort((effort, inherited), config)
        pending_reason = str(pending.get("reason") or "pending task")
        return routed, f"affirmed pending task ({inherited}): {pending_reason}", pending

    # A real new request means the approval question went stale. Drop it so a
    # later bare "yes" cannot accidentally execute old context.
    if _is_substantive_new_request(text):
        _consume_pending_intent(session_id)

    return effort, reason, None


def _max_effort(efforts: Iterable[str], config: dict[str, Any]) -> str:
    best = DEFAULT_CONFIG["default"]
    best_idx = EFFORT_ORDER.index(best)
    for effort in efforts:
        effort = str(effort or "").lower()
        if effort not in EFFORT_ORDER:
            continue
        idx = EFFORT_ORDER.index(effort)
        if idx > best_idx:
            best = effort
            best_idx = idx
    return _clamp_effort(best, config)


def _assistant_asks_to_proceed(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False
    return _matches(_PROCEED_ACTION_PATTERNS, normalized)


def _is_affirmative(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized or len(normalized) > 180:
        return False
    if _is_rejection(normalized):
        return False
    return _matches(_AFFIRMATIVE_PATTERNS, normalized)


def _is_rejection(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return False
    return _matches(_REJECTION_PATTERNS, normalized)


def _is_substantive_new_request(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return False
    if _is_affirmative(normalized) or _is_rejection(normalized):
        return False
    return len(normalized) > 12 or bool(re.search(r"\b(?:what|why|how|when|where|who|patch|run|check|set|create|build|implement|fix|show|list|find)\b", normalized, re.I))


def _session_id_for_key(session_store, session_key: str) -> str:
    if session_store is None or not session_key:
        return ""
    try:
        ensure_loaded = getattr(session_store, "_ensure_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()
    except Exception as exc:
        logger.debug("reasoning-router: session-store load failed: %s", exc)
    try:
        entries = getattr(session_store, "_entries", {})
        entry = entries.get(session_key) if isinstance(entries, dict) else None
        return str(getattr(entry, "session_id", "") or "")
    except Exception as exc:
        logger.debug("reasoning-router: session-id lookup failed: %s", exc)
        return ""


def _pending_intent_ttl_minutes(config: dict[str, Any]) -> int:
    return max(1, _safe_int(config.get("pending_intent_ttl_minutes"), DEFAULT_CONFIG["pending_intent_ttl_minutes"]))


def _active_pending_intent(session_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    pending = _PENDING_INTENTS.get(session_id)
    if not pending:
        return None
    expires_raw = str(pending.get("expires_at") or "")
    try:
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except Exception:
        _PENDING_INTENTS.pop(session_id, None)
        return None
    if datetime.now(timezone.utc) > expires_at or pending.get("consumed"):
        _PENDING_INTENTS.pop(session_id, None)
        return None
    return pending


def _consume_pending_intent(session_id: str) -> None:
    pending = _PENDING_INTENTS.pop(session_id, None)
    if pending is not None:
        pending["consumed"] = True


def _preview(text: str, limit: int = 160) -> str:
    return " ".join(str(text or "").split())[:limit]


def _matches(patterns: Iterable[re.Pattern], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _is_docs_polish_request(text: str) -> bool:
    return all(pattern.search(text) for pattern in _COMPILED_DOCS_POLISH) and not _matches(
        _COMPILED_DOCS_POLISH_RISK,
        text,
    )


def _matched_high_groups(text: str) -> list[str]:
    return [name for name, patterns in _COMPILED_HIGH_GROUPS.items() if _matches(patterns, text)]


def _router_config(gateway) -> dict[str, Any]:
    # Prefer the standalone router config outside the plugin source tree.
    # Legacy plugin-local/main-config fallbacks keep old installs safe during
    # migration, but writes now go to ~/.hermes/reasoning-router/config.yaml.
    merged = _read_router_config_from_disk()
    config = getattr(gateway, "config_data", None)
    if isinstance(config, dict):
        legacy_router = config.get("reasoning_router")
        if isinstance(legacy_router, dict) and not _config_path().exists() and not _legacy_plugin_config_path().exists():
            merged.update(legacy_router)
    if isinstance(_RUNTIME_CONFIG_OVERRIDE, dict):
        merged.update(_RUNTIME_CONFIG_OVERRIDE)
    return merged


def _main_config_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "config.yaml"
    except Exception:
        import os

        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"


def _config_path() -> Path:
    return _hermes_home() / "reasoning-router" / "config.yaml"


def _plugin_config_path() -> Path:
    return _config_path()


def _legacy_plugin_config_path() -> Path:
    return _main_config_path().parent / "plugins" / "reasoning-router" / "config.yaml"


def _hermes_home() -> Path:
    return _main_config_path().parent


def _read_yaml_file(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        logger.warning("reasoning-router: failed to read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _read_full_config() -> dict[str, Any]:
    return _read_yaml_file(_config_path())


def _read_legacy_router_config() -> dict[str, Any]:
    data = _read_yaml_file(_main_config_path())
    router = data.get("reasoning_router")
    return router if isinstance(router, dict) else {}


def _read_router_config_from_disk() -> dict[str, Any]:
    config_path = _config_path()
    router = _read_full_config()
    if not config_path.exists():
        legacy = _read_yaml_file(_legacy_plugin_config_path())
        if not legacy:
            legacy = _read_legacy_router_config()
        if legacy:
            router = legacy
    cfg = {**DEFAULT_CONFIG, **router}
    if isinstance(_RUNTIME_CONFIG_OVERRIDE, dict):
        cfg.update(_RUNTIME_CONFIG_OVERRIDE)
    return cfg


def _update_router_config(updates: dict[str, Any]) -> None:
    global _RUNTIME_CONFIG_OVERRIDE
    if yaml is None:
        raise RuntimeError("PyYAML is required to update reasoning-router config")

    path = _config_path()
    data = _read_router_config_from_disk()
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    current = {**data, **updates}
    _RUNTIME_CONFIG_OVERRIDE = current


def _format_status(config: dict[str, Any]) -> str:
    state = "on" if _truthy(config.get("enabled", True)) else "off"
    pending_state = "on" if _truthy(config.get("pending_intent_enabled", True)) else "off"
    return (
        f"Reasoning router: {state}\n"
        f"config={_config_path()}\n"
        f"min={config.get('min')} default={config.get('default')} max={config.get('max')}\n"
        f"journal_log={bool(_truthy(config.get('log_decisions', True)))} "
        f"decision_log={bool(_truthy(config.get('decision_log', False)))}\n"
        f"decision_log={_decision_log_path(config)}\n"
        f"pending_intent={pending_state} ttl={_pending_intent_ttl_minutes(config)}m active={len(_PENDING_INTENTS)}\n"
        f"xhigh_threshold={_safe_int(config.get('xhigh_high_match_threshold'), DEFAULT_CONFIG['xhigh_high_match_threshold'])} high-complexity categories"
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp_effort(effort: str, config: dict[str, Any]) -> str:
    effort = effort.lower()
    if effort not in EFFORT_ORDER:
        effort = DEFAULT_CONFIG["default"]

    min_effort = str(config.get("min") or DEFAULT_CONFIG["min"]).lower()
    max_effort = str(config.get("max") or DEFAULT_CONFIG["max"]).lower()
    if min_effort not in EFFORT_ORDER:
        min_effort = DEFAULT_CONFIG["min"]
    if max_effort not in EFFORT_ORDER:
        max_effort = DEFAULT_CONFIG["max"]

    idx = EFFORT_ORDER.index(effort)
    min_idx = EFFORT_ORDER.index(min_effort)
    max_idx = EFFORT_ORDER.index(max_effort)
    if min_idx > max_idx:
        min_idx, max_idx = max_idx, min_idx
    idx = max(min_idx, min(max_idx, idx))
    return EFFORT_ORDER[idx]


def _reasoning_config_for_effort(effort: str) -> dict[str, Any]:
    if effort == "none":
        return {"enabled": False}
    return {"enabled": True, "effort": effort}


def _session_key_for(event, gateway, session_store=None) -> str:
    source = getattr(event, "source", None)
    if source is None:
        return ""

    resolver = getattr(gateway, "_session_key_for_source", None)
    if callable(resolver):
        try:
            return str(resolver(source) or "")
        except Exception as exc:
            logger.debug("reasoning-router: gateway session-key lookup failed: %s", exc)

    if session_store is not None:
        generator = getattr(session_store, "_generate_session_key", None)
        if callable(generator):
            try:
                return str(generator(source) or "")
            except Exception as exc:
                logger.debug("reasoning-router: session-store key lookup failed: %s", exc)

    return ""


def _set_reasoning_override(gateway, session_key: str, reasoning_config: dict[str, Any]) -> None:
    setter = getattr(gateway, "_set_session_reasoning_override", None)
    if callable(setter):
        setter(session_key, reasoning_config)
        return

    overrides = getattr(gateway, "_session_reasoning_overrides", None)
    if isinstance(overrides, dict):
        overrides[session_key] = reasoning_config
        return

    raise RuntimeError("gateway does not expose session reasoning overrides")


def _record_decision(
    gateway,
    session_key: str,
    effort: str,
    reason: str,
    text: str,
    *,
    event=None,
    pending_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_key": session_key,
        "platform": platform,
        "user_id": getattr(source, "user_id", None),
        "chat_id": getattr(source, "chat_id", None),
        "thread_id": getattr(source, "thread_id", None),
        "effort": effort,
        "reason": reason,
        "message_preview": text[:160],
    }
    if pending_intent:
        decision["pending_task_preview"] = pending_intent.get("user_preview") or pending_intent.get("assistant_preview")
        decision["pending_assistant_preview"] = pending_intent.get("assistant_preview")
        decision["pending_effort"] = pending_intent.get("effort")

    decisions = getattr(gateway, "_reasoning_router_decisions", None)
    if not isinstance(decisions, dict):
        decisions = {}
        setattr(gateway, "_reasoning_router_decisions", decisions)
    decisions[session_key] = decision
    return decision


def _decision_log_path(config: dict[str, Any]) -> Path:
    raw = str(config.get("decision_log_path") or DEFAULT_CONFIG["decision_log_path"])
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _hermes_home() / path
    return path


def _append_decision_log(config: dict[str, Any], decision: dict[str, Any]) -> None:
    path = _decision_log_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.warning("reasoning-router: failed to append decision log %s: %s", path, exc)


def _format_recent_decisions(config: dict[str, Any], *, limit: int = 5) -> str:
    limit = max(1, min(limit, 20))
    path = _decision_log_path(config)
    if not path.exists():
        return f"No reasoning-router decision log yet at {path}."

    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return f"Could not read reasoning-router decision log at {path}: {exc}"

    if not lines:
        return f"Reasoning-router decision log is empty at {path}."

    rendered = []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
        except Exception:
            rendered.append(f"- malformed row: {line[:120]}")
            continue
        ts = str(row.get("timestamp", ""))[:19]
        effort = row.get("effort", "?")
        reason = row.get("reason", "?")
        preview = str(row.get("message_preview", "")).replace("\n", " ")[:90]
        rendered.append(f"- {ts} {effort}: {reason} — {preview}")

    return "Recent reasoning-router decisions:\n" + "\n".join(rendered)
