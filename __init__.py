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
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - Hermes normally depends on PyYAML
    yaml = None

logger = logging.getLogger(__name__)

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"ultra": "max"}
DEFAULT_CONFIG = {
    "enabled": True,
    "default": "medium",
    "min": "none",
    "max": "xhigh",
    "shadow_mode": False,
    # Chat surfaces the router is allowed to affect. Unsupported platforms fail
    # open without mutating session reasoning.
    "enabled_platforms": ["discord", "telegram", "buzz"],
    # systemd/journald logging through the normal Hermes gateway logger
    "log_decisions": True,
    # persistent JSONL audit trail for later inspection or external audits
    "decision_log": False,
    "decision_log_path": "logs/reasoning-router.jsonl",
    "low_char_limit": 80,
    "xhigh_high_match_threshold": 4,
    # Disabled-by-default POC: an isolated codex-proxy classifier can raise
    # ambiguous low/default deterministic routes without replacing guardrails.
    "semantic_classifier_enabled": False,
    "semantic_classifier_url": "http://127.0.0.1:8080/v1/chat/completions",
    "semantic_classifier_model": "gpt-5.6-luna",
    "semantic_classifier_api_key": "",
    "semantic_classifier_timeout_seconds": 8,
    "semantic_classifier_min_confidence": 0.75,
    "semantic_classifier_max_chars": 1200,
    # Carry task complexity across terse approvals like "yes" / "go ahead".
    "pending_intent_enabled": True,
    "pending_intent_ttl_minutes": 30,
}

# ``Ultra`` is Codex's orchestration label, not a provider reasoning.effort wire
# value. Normalize explicit Ultra requests to the canonical ``max`` effort.
_MAX_PATTERNS = (
    r"^(?:please\s+)?(?:use|set|switch|route|run|enable|apply)\s+(?:the\s+)?(?:max(?:imum)?|ultra)(?:\s+reasoning)?\b",
    r"^(?:can|could|would|will)\s+you(?:\s+please)?\s+(?:use|set|switch|route|run|enable|apply)\s+(?:the\s+)?(?:max(?:imum)?|ultra)(?:\s+reasoning)?\b",
    r"^(?:please\s+)?(?:reason|think)\b.{0,24}\b(?:at\s+)?(?:max(?:imum)?|ultra)\b",
    r"^(?:max(?:imum)?\s+reasoning|ultra(?:\s+reasoning)?)(?:\s*,?\s*please)?(?:\s+(?:for|on|to)\b.*)?[.!?\s]*$",
)

# Explicit xhigh means: slow down; this is multi-system, risky, architectural,
# security-sensitive, or asks for unusually complete execution.
_XHIGH_PATTERNS = (
    r"\b(xhigh|extra\s*high|think\s+hard(?:er)?)\b",
    r"\b(be\s+thorough|flesh\s+out|boil\s+the\s+ocean|do\s+the\s+whole\s+thing|end\s+to\s+end)\b",
    r"\b(architecture|architectural|design\s+decision|tradeoff|strategy|migration\s+plan)\b",
    r"\b(security|auth|oauth|credential|secret|permission|token|ssrf|injection)\b",
    r"\b(?:production\s+(?:system|service|deploy|deployment|incident|outage)|prod\s+(?:deploy|deployment|incident|outage)|rollback\s+safety|rollback-safe|data\s+loss|incident|outage)\b",
    r"\b(?:back\s*up|backup|rollback|restore)\b.{0,200}\b(?:all\s+of\s+them|all\s+(?:current\s+)?skills?|all\s+(?:files?|docs?|references?|pages?)|every|entire|whole|bulk)\b.{0,200}\b(?:remove|delete|scrub|purge|strip)\b",
    r"\b(?:all\s+of\s+them|all\s+(?:current\s+)?skills?|all\s+(?:files?|docs?|references?|pages?)|every|entire|whole|bulk|back\s*up|backup|rollback|restore)\b.{0,200}\b(?:remove|delete|scrub|purge|strip)\b.{0,120}\b(?:any|all|every)\s+(?:mention|mentions|reference|references|occurrence|occurrences)\b",
    r"\b(?:shut\s*down|shutdown|stop|disable|restart)\b.{0,120}\b(?:mcp|gateway|daemon|systemd|service|container|docker|postgres|redis|api|worker|server)\b",
    r"\b(?:mcp|gateway|daemon|systemd|service|container|docker|postgres|redis|api|worker|server)\b.{0,120}\b(?:shut\s*down|shutdown|stop|disable|restart)\b",
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
    r"\b(overly\s+descriptive|wording|generic|standard[is]ed|style|mention|document|macos|linux|systemctl|restart\s+instructions?|need\s+to\s+restart|restart\s+the\s+gateway|ask\s+the\s+user\s+to\s+restart)\b",
)

_DOCS_POLISH_RISK_PATTERNS = (
    r"\b(security|auth|oauth|credential|secret|permission|token|ssrf|injection)\b",
    r"\b(?:production\s+(?:system|service|deploy|deployment|incident|outage)|prod\s+(?:deploy|deployment|incident|outage)|deploy|deployment|rollback\s+safety|rollback-safe|data\s+loss|incident|outage)\b",
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
        r"\b(systemd|service|restart|deploy|auth|oauth|credential|production\s+(?:system|service|deploy|deployment|incident|outage))\b",
    ),
    "real_world_device_schedule": (
        r"\b(?:turn\s+on|turn\s+off|start|stop|open|close|lock|unlock|set)\b.{0,160}\b(?:pump|pool|heater|chlorinator|salt\s+generator|lights?|switch|fan|sprinkler|valve|cover|door|garage|lock|thermostat|climate)\b.{0,160}\b(?:in\s+\d+|after\s+\d+|for\s+\d+|again|schedule|timer|later|at\s+\d{1,2})\b",
        r"\b(?:pump|pool|heater|chlorinator|salt\s+generator|lights?|switch|fan|sprinkler|valve|cover|door|garage|lock|thermostat|climate)\b.{0,160}\b(?:turn\s+on|turn\s+off|start|stop|open|close|lock|unlock|set)\b.{0,160}\b(?:in\s+\d+|after\s+\d+|for\s+\d+|again|schedule|timer|later|at\s+\d{1,2})\b",
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
    "github workflow": (
        r"\b(?:make|open|create)\s+(?:a\s+)?(?:pr|prs|pr's|pull\s+requests?)\b.{0,80}\bmerge\b",
        r"\b(?:pr|prs|pr's|pull\s+requests?)\b.{0,80}\bmerge\b",
        r"\bmerge\b.{0,80}\b(?:pr|prs|pr's|pull\s+requests?)\b",
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

_NONE_PATTERNS = (
    r"^(?:lol|haha|lmao|heh|thanks|thank\s+you|ok|okay|cool|nice)[.!?\s]*$",
    r"^(?:what\s+time\s+is\s+it|what\s+date\s+is\s+it|what'?s\s+(?:today'?s\s+)?date)[?!.\s]*$",
)

_LOW_PATTERNS = (
    r"^(?:lol|haha|lmao|heh)?\s*(thanks|thank you|ok|okay|yes|no|yep|nope|cool|nice)[.!?\s]*$",
    r"\b(what\s+time|what\s+date|who\s+is|what\s+is)\b",
    r"\b(quick|brief|one\s+sentence|short answer)\b",
)

_COMPILED_MAX = tuple(re.compile(pattern, re.I) for pattern in _MAX_PATTERNS)
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
_COMPILED_NONE = tuple(re.compile(pattern, re.I) for pattern in _NONE_PATTERNS)
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
_LAST_HEALTH: dict[str, dict[str, Any] | None] = {
    "route": None,
    "override": None,
    "decision_log": None,
}


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_command(
        "reasoning-router",
        reasoning_router_command,
        description="Toggle/status/configure automatic reasoning effort routing",
        args_hint="status|on|off|min|max|default|threshold|pending|platforms|shadow|log|recent|test <message>",
    )


def pre_gateway_dispatch(event=None, gateway=None, session_store=None, **_kwargs):
    """Route a gateway message to a reasoning effort.

    Return shape follows Hermes' pre_gateway_dispatch contract. We never skip or
    rewrite user messages; the plugin only mutates the gateway's per-session
    reasoning override before normal dispatch continues.
    """
    if event is None or gateway is None:
        return None

    if bool(getattr(event, "internal", False)):
        return None

    # Gateway dispatch runs plugin hooks before central authorization. Avoid
    # creating overrides, pending-intent changes, or logs for rejected senders.
    auth_fn = getattr(gateway, "_is_user_authorized", None)
    if callable(auth_fn):
        try:
            if not auth_fn(getattr(event, "source", None)):
                return None
        except Exception:
            logger.debug("reasoning-router: authorization check failed", exc_info=True)
            return None

    text = str(getattr(event, "text", "") or "")
    if not text.strip():
        return None

    # Built-in/plugin slash commands should keep their own semantics. In
    # particular, /reasoning must be able to set manual state without us racing
    # it from the pre-dispatch hook.
    if text.lstrip().startswith("/"):
        if not _slash_command_preserves_pending(text):
            _consume_pending_intent_for_event(event, gateway, session_store)
        return None

    config = _router_config(gateway)
    if not _truthy(config.get("enabled", True)):
        return None

    if not _platform_enabled(event, config):
        logger.debug("reasoning-router: platform not enabled; allowing without override")
        return None

    session_key = _session_key_for(event, gateway, session_store)
    if not session_key:
        logger.debug("reasoning-router: no session key; allowing without override")
        return None

    effort, reason, pending_intent = _effective_effort_for_message(
        text,
        config,
        session_store=session_store,
        session_key=session_key,
    )
    reasoning_config = _reasoning_config_for_effort(effort)
    route_metadata = _route_metadata_for_decision(
        effort,
        reason,
        text,
        config,
        pending_intent=pending_intent,
    )
    shadow_mode = _truthy(config.get("shadow_mode", False))

    if shadow_mode:
        override_applied = False
        _record_health(
            "override",
            status="shadow",
            session_key=session_key,
            effort=effort,
        )
    else:
        try:
            _set_reasoning_override(gateway, session_key, reasoning_config)
        except Exception as exc:
            logger.warning("reasoning-router: failed to set session reasoning override: %s", exc)
            _record_health(
                "override",
                status="failed",
                session_key=session_key,
                effort=effort,
                error=str(exc),
            )
            return None
        override_applied = True
        _record_health(
            "override",
            status="applied",
            session_key=session_key,
            effort=effort,
        )

    decision = _record_decision(
        gateway,
        session_key,
        effort,
        reason,
        text,
        event=event,
        pending_intent=pending_intent,
        route_metadata=route_metadata,
        shadow_mode=shadow_mode,
        override_applied=override_applied,
    )
    _record_health(
        "route",
        status="ok",
        session_key=session_key,
        effort=effort,
        reason=reason,
        shadow_mode=shadow_mode,
    )

    if _truthy(config.get("log_decisions", True)):
        logger.info(
            "reasoning-router: session=%s effort=%s reason=%s",
            session_key,
            effort,
            reason,
        )

    if _truthy(config.get("decision_log", False)):
        ok, error = _append_decision_log(config, decision)
        _record_health(
            "decision_log",
            status="ok" if ok else "failed",
            path=str(_decision_log_path(config)),
            error=error,
        )
    else:
        _record_health("decision_log", status="disabled")

    return None


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
            "default <effort>|threshold <N>|pending [status|clear|on|off]|"
            "platforms [list]|shadow on|off|log on|off|recent [N]|test <message>`\n"
            "Efforts: none, minimal, low, medium, high, xhigh, max (Ultra is accepted as an alias for max)."
        )

    if command in {"on", "enable", "enabled"}:
        _update_router_config({"enabled": True})
        return "Reasoning router enabled."

    if command in {"off", "disable", "disabled"}:
        _update_router_config({"enabled": False})
        return "Reasoning router disabled. Use `/reasoning-router on` to re-enable."

    if command in {"min", "max", "default"}:
        effort = _normalize_effort_name(value)
        if effort not in EFFORT_ORDER:
            return f"Invalid effort `{value}`. Use one of: {', '.join(EFFORT_ORDER)}."
        _update_router_config({command: effort})
        return f"Reasoning router {command} effort set to {effort}."

    if command in {"shadow", "shadow-mode"}:
        lowered = value.lower()
        if lowered in {"on", "enable", "enabled", "true", "1", "yes"}:
            _update_router_config({"shadow_mode": True})
            return "Reasoning router shadow mode enabled."
        if lowered in {"off", "disable", "disabled", "false", "0", "no"}:
            _update_router_config({"shadow_mode": False})
            return "Reasoning router shadow mode disabled."
        return "Usage: `/reasoning-router shadow on|off`"

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

    if command in {"platform", "platforms", "enabled-platforms"}:
        cfg = _read_router_config_from_disk()
        if not value:
            return _format_platforms_status(cfg)
        platforms = _parse_platform_values(value)
        if not platforms:
            return "Usage: `/reasoning-router platforms discord,telegram,buzz|all`"
        _update_router_config({"enabled_platforms": platforms})
        return f"Reasoning router enabled platforms set to: {', '.join(platforms)}."

    if command in {"pending", "continuation", "latch"}:
        lowered = value.lower()
        if not lowered or lowered == "status":
            return _format_pending_status()
        if lowered == "clear":
            count = len(_PENDING_INTENTS)
            _PENDING_INTENTS.clear()
            return f"Reasoning router pending intents cleared ({count})."
        if lowered in {"on", "enable", "enabled", "true", "1", "yes"}:
            _update_router_config({"pending_intent_enabled": True})
            return "Reasoning router pending-intent inheritance enabled."
        if lowered in {"off", "disable", "disabled", "false", "0", "no"}:
            _update_router_config({"pending_intent_enabled": False})
            _PENDING_INTENTS.clear()
            return "Reasoning router pending-intent inheritance disabled and cleared."
        return "Usage: `/reasoning-router pending status|clear|on|off`"

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

    # Strongest wins. Avoid low-routing a short sentence like "go ahead and set
    # up the automation" just because it is brief. Definition-style questions
    # get a cheap factual route before risk keywords so "what is OAuth?" does
    # not look like an auth migration.
    if _is_simple_factual_question(lowered):
        return _clamp_effort("low", cfg), "simple factual question"

    # Explicit effort directives outrank content-category shortcuts: "Use maximum
    # reasoning to polish the README" is still a request to use maximum effort.
    if _matches(_COMPILED_MAX, lowered):
        return _clamp_effort("max", cfg), "explicit maximum/Ultra reasoning request"

    # Documentation/install-prompt wording can mention operational words like
    # "restart the gateway" or "systemctl" without asking us to touch live ops.
    # Keep that at medium unless other non-docs risk categories dominate.
    if _is_docs_polish_request(lowered):
        return _clamp_effort("medium", cfg), "documentation wording/install-prompt polish"

    # Question/clarification forms get one semantic pass so words like
    # "restart gateway" do not over-route when the user is only asking whether a
    # restart is needed.
    if _matches(_COMPILED_XHIGH, lowered):
        if _is_question_or_clarification(lowered):
            semantic_route = _semantic_route_for_ambiguous_message(
                text,
                lowered,
                cfg,
                baseline_effort="xhigh",
                baseline_reason="matched xhigh complexity/risk keywords",
                allow_lowering=True,
            )
            if semantic_route is not None:
                return semantic_route
        return _clamp_effort("xhigh", cfg), "matched xhigh complexity/risk keywords"

    if _matches(_COMPILED_IMPLEMENTATION_APPROVAL, lowered):
        return _clamp_effort("high", cfg), "matched implementation approval/tweak request"

    high_groups = _matched_high_groups(lowered)
    threshold = _safe_int(cfg.get("xhigh_high_match_threshold"), DEFAULT_CONFIG["xhigh_high_match_threshold"])
    if len(high_groups) >= threshold:
        baseline_reason = f"matched multiple high-complexity categories: {', '.join(high_groups)}"
        if _is_question_or_clarification(lowered):
            semantic_route = _semantic_route_for_ambiguous_message(
                text,
                lowered,
                cfg,
                baseline_effort="xhigh",
                baseline_reason=baseline_reason,
                allow_lowering=True,
            )
            if semantic_route is not None:
                return semantic_route
        return (
            _clamp_effort("xhigh", cfg),
            baseline_reason,
        )

    if high_groups:
        baseline_reason = f"matched high-complexity category: {', '.join(high_groups)}"
        if _is_question_or_clarification(lowered):
            semantic_route = _semantic_route_for_ambiguous_message(
                text,
                lowered,
                cfg,
                baseline_effort="high",
                baseline_reason=baseline_reason,
                allow_lowering=True,
            )
            if semantic_route is not None:
                return semantic_route
        return (
            _clamp_effort("high", cfg),
            baseline_reason,
        )

    if _matches(_COMPILED_MEDIUM_TECH_FOLLOWUP, lowered):
        return _clamp_effort("medium", cfg), "matched technical feasibility/design follow-up"

    if _matches(_COMPILED_MEDIUM_OPINION, lowered):
        return _clamp_effort("medium", cfg), "matched opinion/take request"

    if _is_explicit_config_snippet_request(text):
        return _clamp_effort("medium", cfg), "matched explicit config snippet request"

    if _matches(_COMPILED_NONE, lowered):
        return _clamp_effort("none", cfg), "matched no-op/simple time-date request"

    semantic_route = _semantic_route_for_ambiguous_message(text, lowered, cfg)
    if semantic_route is not None:
        return semantic_route

    if _matches(_COMPILED_LOW, lowered) or len(normalized) <= _safe_int(cfg.get("low_char_limit"), 80):
        return _clamp_effort("low", cfg), "quick/simple message"

    if _matches(_COMPILED_MEDIUM, lowered):
        return _clamp_effort("medium", cfg), "matched normal tool/status keywords"

    default = _normalize_effort_name(cfg.get("default") or DEFAULT_CONFIG["default"])
    if default not in EFFORT_ORDER:
        default = DEFAULT_CONFIG["default"]
    return _clamp_effort(default, cfg), "default route"


def _semantic_route_for_ambiguous_message(
    text: str,
    lowered: str,
    config: dict[str, Any],
    *,
    baseline_effort: str = "low",
    baseline_reason: str = "quick/simple message",
    allow_lowering: bool = False,
) -> tuple[str, str] | None:
    """Optionally adjust an ambiguous deterministic route with an isolated classifier.

    The POC is deliberately conservative: it is disabled by default, skips
    obvious low chatter, raises low/default routes, and lowers high/xhigh only
    for question/clarification forms that look like deterministic false positives.
    """
    if not _truthy(config.get("semantic_classifier_enabled", False)):
        return None

    normalized = " ".join(str(text or "").split())
    if not normalized:
        return None
    if len(normalized) > _safe_int(
        config.get("semantic_classifier_max_chars"),
        DEFAULT_CONFIG["semantic_classifier_max_chars"],
    ):
        return None

    # These are cheap, explicit low signals. Length alone is not treated as
    # obvious-low because short imperatives like "Set this one please" are the
    # exact ambiguous class this POC is meant to catch.
    if _matches(_COMPILED_LOW, lowered) and not _is_question_or_clarification(lowered):
        return None

    try:
        result = _semantic_classify_with_codex_proxy(text, config)
    except Exception as exc:
        logger.warning("reasoning-router: semantic classifier failed: %s", exc)
        return None

    parsed = _normalize_semantic_classifier_result(result)
    if not parsed:
        return None

    effort = str(parsed.get("effort") or "").lower()
    confidence = float(parsed.get("confidence") or 0.0)
    min_confidence = _safe_float(
        config.get("semantic_classifier_min_confidence"),
        DEFAULT_CONFIG["semantic_classifier_min_confidence"],
    )
    min_confidence = max(0.0, min(1.0, min_confidence))
    if confidence < min_confidence:
        return None

    baseline = _clamp_effort(baseline_effort, config)
    routed = _clamp_effort(effort, config)
    baseline_idx = EFFORT_ORDER.index(baseline)
    routed_idx = EFFORT_ORDER.index(routed)

    categories = parsed.get("risk_categories") or []
    if isinstance(categories, str):
        categories = [categories]
    category_text = ", ".join(str(item) for item in categories if str(item).strip())
    reason = str(parsed.get("reason") or "semantic classification").strip()
    if category_text:
        reason = f"{reason}; categories: {category_text}"

    if routed_idx > baseline_idx:
        return routed, f"semantic classifier raised ambiguous route to {routed}: {reason}"
    if allow_lowering and routed_idx < baseline_idx and routed_idx >= EFFORT_ORDER.index("medium"):
        return routed, f"semantic classifier lowered question/clarification route from {baseline}: {reason}"
    return None


def _normalize_semantic_classifier_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, str):
        try:
            result = json.loads(_extract_json_object(result))
        except Exception:
            return None
    if not isinstance(result, dict):
        return None

    effort = _normalize_effort_name(result.get("effort") or "")
    if effort not in EFFORT_ORDER:
        return None
    if effort == "minimal":
        effort = "low"

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        return None
    confidence = max(0.0, min(1.0, confidence))

    categories = result.get("risk_categories") or []
    if not isinstance(categories, list):
        categories = [str(categories)]

    return {
        "effort": effort,
        "confidence": confidence,
        "risk_categories": categories[:8],
        "reason": str(result.get("reason") or "").strip()[:240],
    }


def _extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return raw[start : end + 1]


def _semantic_classifier_messages(
    text: str,
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    ctx = context if isinstance(context, dict) else {}
    payload = {
        "current_user_message": str(text or "")[:1200],
        "last_assistant_intent": str(ctx.get("last_assistant_intent") or "")[:500],
        "pending_action": str(ctx.get("pending_action") or "")[:120],
        "recent_messages": _compact_recent_messages(ctx.get("recent_messages")),
    }
    system = (
        "You are a routing classifier. Return JSON only. "
        "Classify the user request into exactly one effort: none, low, medium, high, xhigh. "
        "Use last_assistant_intent and recent_messages only to resolve terse approvals, deictic references, and pending action scope. "
        "Treat terse deictic imperatives that ask to set, change, apply, use, or do this/that/it as medium unless they are clearly casual chatter. "
        "none: pure acknowledgements, thanks, laughter, casual comments, or simple time/date questions that require no judgment, no memory, no action planning, and no side effects. "
        "low: simple factual/status replies with minimal judgment and no meaningful risk. "
        "medium: explanation, research, inspection, simple reversible action, or one setting value change. "
        "high: coding, config changes with files/tests, debugging, verification-heavy work, GitHub workflow, or real-world device actions with scheduling/follow-up. "
        "xhigh: architecture, multi-system changes, migrations, auth/security/secrets, destructive or rollback-sensitive actions, service restarts, deploys, or production-system incidents. "
        "Return keys: effort, confidence, risk_categories, reason. "
        "confidence must be a number from 0 to 1. Do not solve the request."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _compact_recent_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("summary") or item.get("content") or item.get("text") or "")
        content = " ".join(content.split())[:500]
        if not content:
            continue
        compact.append({"role": role, "content": content})
    return compact[-3:]


def _is_question_or_clarification(lowered: str) -> bool:
    text = str(lowered or "").strip()
    if not text:
        return False
    if "?" in text:
        return True
    return bool(
        re.search(
            r"\b(?:do you need|should i|should we|would we|are you saying|so you(?:'re| are) saying|is this|does this|would this|can this|could this)\b",
            text,
            re.I,
        )
    )


def _semantic_classify_with_codex_proxy(text: str, config: dict[str, Any]) -> dict[str, Any] | None:
    url = str(config.get("semantic_classifier_url") or DEFAULT_CONFIG["semantic_classifier_url"])
    model = str(config.get("semantic_classifier_model") or DEFAULT_CONFIG["semantic_classifier_model"])
    configured_api_key = str(config.get("semantic_classifier_api_key") or "").strip()
    api_key = str(
        configured_api_key
        or os.environ.get("CODEX_PROXY_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or DEFAULT_CONFIG["semantic_classifier_api_key"]
    ).strip()
    timeout = max(
        1,
        _safe_int(
            config.get("semantic_classifier_timeout_seconds"),
            DEFAULT_CONFIG["semantic_classifier_timeout_seconds"],
        ),
    )
    payload = {
        "model": model,
        "messages": _semantic_classifier_messages(text, config),
        "max_tokens": 220,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str) or not content.strip():
        return None
    return _normalize_semantic_classifier_result(content)


def _effective_effort_for_message(
    text: str,
    config: dict[str, Any],
    *,
    session_store=None,
    session_key: str = "",
) -> tuple[str, str, dict[str, Any] | None]:
    session_id = _session_id_for_key(session_store, session_key)
    route_config = dict(config)
    if session_id:
        recent_messages = _recent_messages_for_session(session_id)
        if recent_messages:
            route_config["recent_messages"] = recent_messages
            last_assistant = next(
                (item.get("content", "") for item in reversed(recent_messages) if item.get("role") == "assistant"),
                "",
            )
            if last_assistant and not route_config.get("last_assistant_intent"):
                route_config["last_assistant_intent"] = last_assistant[:500]

    effort, reason = classify_message(text, route_config)
    if not _truthy(config.get("pending_intent_enabled", True)):
        return effort, reason, None

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

    # Pending intents are one-shot. Any non-empty user turn that is not the
    # affirmative consumes the pending task so a later bare "yes" cannot inherit
    # stale context.
    _consume_pending_intent(session_id)
    return effort, f"cleared pending task; {reason}", pending


def _recent_messages_for_session(session_id: str, limit: int = 3) -> list[dict[str, str]]:
    sid = str(session_id or "").strip()
    if not sid:
        return []
    db_path = _hermes_home() / "state.db"
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0) as con:
            rows = list(
                con.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                      AND role IN ('user', 'assistant')
                      AND COALESCE(content, '') != ''
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (sid, max(limit * 3, limit)),
                )
            )
    except Exception as exc:
        logger.debug("reasoning-router: failed to read recent session messages: %s", exc)
        return []

    recent: list[dict[str, str]] = []
    for role, content in reversed(rows):
        compact = " ".join(str(content or "").split())[:500]
        if compact:
            recent.append({"role": str(role), "content": compact})
    return recent[-limit:]


def _max_effort(efforts: Iterable[str], config: dict[str, Any]) -> str:
    best = "none"
    best_idx = EFFORT_ORDER.index(best)
    for effort in efforts:
        effort = _normalize_effort_name(effort)
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


def _consume_pending_intent_for_event(event, gateway, session_store=None) -> None:
    session_key = _session_key_for(event, gateway, session_store)
    if not session_key:
        return
    session_id = _session_id_for_key(session_store, session_key)
    if session_id:
        _consume_pending_intent(session_id)


def _consume_pending_intent(session_id: str) -> None:
    pending = _PENDING_INTENTS.pop(session_id, None)
    if pending is not None:
        pending["consumed"] = True


def _is_simple_factual_question(lowered: str) -> bool:
    text = str(lowered or "").strip()
    if not text:
        return False
    if not re.match(r"^(?:what|who|when|where)\s+(?:is|are|was|were)\b", text):
        return False
    if re.search(r"\b(?:fix|debug|implement|build|change|modify|configure|deploy|restart|delete|remove|migrate|patch|update|review|audit|secure|rotate)\b", text):
        return False
    return len(text) <= 120


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


def _is_explicit_config_snippet_request(text: str) -> bool:
    raw = str(text or "")
    if "\n" not in raw and "\r" not in raw:
        return False

    first_line = raw.splitlines()[0] if raw.splitlines() else ""
    if not re.search(r"\b(?:set|configure|change|apply|use)\b", first_line, re.I):
        return False
    if not re.search(r"\bconfigure\b", first_line, re.I) and not re.search(
        r"\b(?:this|that|it|config|configuration|setting|value)\b",
        first_line,
        re.I,
    ):
        return False

    if re.search(r":\s*\r?\n\s*$", raw):
        return True

    return bool(
        re.search(r"```(?:yaml|yml)?\s*\r?\n", raw, re.I)
        or re.search(r"(?m)^\s*[-\w.\"']+\s*:\s*\S+", raw)
    )


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


def _read_router_config_from_disk(*, include_runtime_override: bool = True) -> dict[str, Any]:
    config_path = _config_path()
    router = _read_full_config()
    if not config_path.exists():
        legacy = _read_yaml_file(_legacy_plugin_config_path())
        if not legacy:
            legacy = _read_legacy_router_config()
        if legacy:
            router = legacy
    cfg = {**DEFAULT_CONFIG, **router}
    if include_runtime_override and isinstance(_RUNTIME_CONFIG_OVERRIDE, dict):
        cfg.update(_RUNTIME_CONFIG_OVERRIDE)
    return cfg


def _update_router_config(updates: dict[str, Any]) -> None:
    global _RUNTIME_CONFIG_OVERRIDE
    if yaml is None:
        raise RuntimeError("PyYAML is required to update reasoning-router config")

    path = _config_path()
    data = _read_router_config_from_disk(include_runtime_override=False)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    _RUNTIME_CONFIG_OVERRIDE = None


def _format_status(config: dict[str, Any]) -> str:
    state = "on" if _truthy(config.get("enabled", True)) else "off"
    pending_state = "on" if _truthy(config.get("pending_intent_enabled", True)) else "off"
    active_pending = _active_pending_intent_count()
    return (
        f"Reasoning router: {state}\n"
        f"config={_config_path()}\n"
        f"min={config.get('min')} default={config.get('default')} max={config.get('max')} "
        f"shadow_mode={'on' if _truthy(config.get('shadow_mode', False)) else 'off'}\n"
        f"platforms={', '.join(sorted(_enabled_platform_names(config)))}\n"
        f"journal_log={bool(_truthy(config.get('log_decisions', True)))} "
        f"decision_log={bool(_truthy(config.get('decision_log', False)))}\n"
        f"decision_log={_decision_log_path(config)}\n"
        f"pending_intent={pending_state} ttl={_pending_intent_ttl_minutes(config)}m active={active_pending}\n"
        f"xhigh_threshold={_safe_int(config.get('xhigh_high_match_threshold'), DEFAULT_CONFIG['xhigh_high_match_threshold'])} high-complexity categories\n"
        f"{_format_health_status()}"
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _platform_name(event) -> str:
    source = getattr(event, "source", None)
    platform = getattr(source, "platform", None)
    value = getattr(platform, "value", platform)
    return str(value or "").strip().lower()


def _enabled_platform_names(config: dict[str, Any]) -> set[str]:
    raw = config.get("enabled_platforms", DEFAULT_CONFIG["enabled_platforms"])
    if isinstance(raw, str):
        values = re.split(r"[,\s]+", raw)
    elif isinstance(raw, Iterable):
        values = raw
    else:
        values = DEFAULT_CONFIG["enabled_platforms"]
    return {str(value).strip().lower() for value in values if str(value).strip()}

def _parse_platform_values(value: str) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,\s]+", value):
        item = part.strip().lower()
        if item and item not in seen:
            parsed.append(item)
            seen.add(item)
    return parsed


def _format_platforms_status(config: dict[str, Any]) -> str:
    enabled = sorted(_enabled_platform_names(config))
    return "Reasoning router enabled platforms: " + (", ".join(enabled) if enabled else "(none)")


def _slash_command_preserves_pending(text: str) -> bool:
    parts = str(text or "").strip().split()
    if not parts:
        return False
    command = parts[0].lower()
    if command != "/reasoning-router":
        return False
    return len(parts) <= 1 or (len(parts) >= 2 and parts[1].lower() == "pending" and (len(parts) == 2 or parts[2].lower() == "status"))


def _active_pending_intent_count() -> int:
    return sum(
        1
        for session_id in list(_PENDING_INTENTS)
        if _active_pending_intent(session_id, DEFAULT_CONFIG)
    )


def _format_pending_status() -> str:
    active = [
        pending
        for session_id in list(_PENDING_INTENTS)
        for pending in [_active_pending_intent(session_id, DEFAULT_CONFIG)]
        if pending
    ]
    if not active:
        return "No active reasoning-router pending intents."
    rendered = []
    for pending in active[:5]:
        effort = pending.get("effort", "?")
        expires = str(pending.get("expires_at", ""))[:19]
        preview = str(pending.get("user_preview") or pending.get("assistant_preview") or "")[:90]
        rendered.append(f"- session={pending.get('session_id', '?')} effort={effort} expires={expires} — {preview}")
    suffix = "" if len(active) <= 5 else f"\n... {len(active) - 5} more"
    return "Active reasoning-router pending intents:\n" + "\n".join(rendered) + suffix


def _record_health(section: str, **values: Any) -> None:
    if section not in _LAST_HEALTH:
        return
    _LAST_HEALTH[section] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **values,
    }


def _format_health_status() -> str:
    route = _LAST_HEALTH.get("route") or {}
    override = _LAST_HEALTH.get("override") or {}
    decision_log = _LAST_HEALTH.get("decision_log") or {}

    route_text = "none"
    if route:
        route_text = (
            f"{str(route.get('timestamp', ''))[:19]} "
            f"session={route.get('session_key', '?')} effort={route.get('effort', '?')}"
        )

    override_text = str(override.get("status", "none"))
    if override.get("error"):
        override_text += f" error={str(override.get('error'))[:80]}"

    log_text = str(decision_log.get("status", "none"))
    if decision_log.get("error"):
        log_text += f" error={str(decision_log.get('error'))[:80]}"

    return (
        f"last_route={route_text}\n"
        f"last_override={override_text}\n"
        f"last_decision_log={log_text}"
    )



def _platform_enabled(event, config: dict[str, Any]) -> bool:
    platform = _platform_name(event)
    if not platform:
        return False
    enabled = _enabled_platform_names(config)
    return "*" in enabled or "all" in enabled or platform in enabled


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_effort_name(value: Any) -> str:
    effort = str(value or "").strip().lower()
    return EFFORT_ALIASES.get(effort, effort)


def _clamp_effort(effort: str, config: dict[str, Any]) -> str:
    effort = _normalize_effort_name(effort)
    if effort not in EFFORT_ORDER:
        effort = DEFAULT_CONFIG["default"]

    min_effort = _normalize_effort_name(config.get("min") or DEFAULT_CONFIG["min"])
    max_effort = _normalize_effort_name(config.get("max") or DEFAULT_CONFIG["max"])
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
    effort = _normalize_effort_name(effort)
    if effort == "none":
        return {"enabled": False}
    if effort not in EFFORT_ORDER:
        effort = DEFAULT_CONFIG["default"]
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


def _route_metadata_for_decision(
    effort: str,
    reason: str,
    text: str,
    config: dict[str, Any],
    *,
    pending_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lowered = " ".join(str(text or "").split()).lower()
    matched_groups = _matched_high_groups(lowered)
    route_source = "deterministic"
    route_detail = "deterministic"
    if "semantic classifier" in reason:
        route_source = "semantic"
        route_detail = "semantic"
    elif pending_intent:
        route_source = "pending"
        if reason.startswith("affirmed pending"):
            route_detail = "pending_affirmed"
        elif reason.startswith("rejected pending"):
            route_detail = "pending_rejected"
        else:
            route_detail = "pending_cleared"
    elif reason == "default route":
        route_source = "default"
        route_detail = "default"
    elif matched_groups:
        route_detail = "high_groups"
    elif reason == "simple factual question":
        route_detail = "simple_factual"
    elif reason == "quick/simple message":
        route_detail = "quick_simple"
    elif reason.startswith("matched no-op"):
        route_detail = "no_op"

    raw_effort = _raw_effort_for_reason(reason, config)
    clamped_from = raw_effort if raw_effort and raw_effort != effort else None
    metadata: dict[str, Any] = {
        "route_source": route_source,
        "route_detail": route_detail,
        "matched_groups": matched_groups,
        "clamped_from": clamped_from,
    }
    if pending_intent:
        metadata["pending_intent_used"] = reason.startswith("affirmed pending")
        metadata["pending_intent_cleared"] = True
    return metadata


def _raw_effort_for_reason(reason: str, config: dict[str, Any]) -> str | None:
    if reason.startswith("affirmed pending"):
        match = re.search(r"\(([^)]+)\)", reason)
        return match.group(1) if match else None
    if "xhigh" in reason or "multiple high-complexity" in reason:
        return "xhigh"
    if "high-complexity category" in reason or "implementation approval" in reason:
        return "high"
    if (
        "documentation wording" in reason
        or "technical feasibility" in reason
        or "opinion" in reason
        or "config snippet" in reason
        or "normal tool/status" in reason
        or "semantic classifier" in reason
    ):
        return "medium"
    if "no-op" in reason:
        return "none"
    if reason in {"simple factual question", "quick/simple message"}:
        return "low"
    if reason == "default route":
        default = str(config.get("default") or DEFAULT_CONFIG["default"]).lower()
        return default if default in EFFORT_ORDER else DEFAULT_CONFIG["default"]
    return None


def _record_decision(
    gateway,
    session_key: str,
    effort: str,
    reason: str,
    text: str,
    *,
    event=None,
    pending_intent: dict[str, Any] | None = None,
    route_metadata: dict[str, Any] | None = None,
    shadow_mode: bool = False,
    override_applied: bool = True,
) -> dict[str, Any]:
    source = getattr(event, "source", None)
    platform = _platform_name(event)
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
        "shadow_mode": bool(shadow_mode),
        "override_applied": bool(override_applied),
        **(route_metadata or {}),
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


def _append_decision_log(config: dict[str, Any], decision: dict[str, Any]) -> tuple[bool, str | None]:
    path = _decision_log_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\n")
        return True, None
    except Exception as exc:
        logger.warning("reasoning-router: failed to append decision log %s: %s", path, exc)
        return False, str(exc)


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
