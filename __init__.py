"""Automatic reasoning-effort router for Hermes gateway sessions.

This plugin intentionally does not patch Hermes core. It uses the existing
``pre_gateway_dispatch`` hook and the gateway's session-scoped reasoning
override mechanism. The gateway later resolves that override and sets
``agent.reasoning_config`` before the provider request is built, so this changes
the real backend reasoning parameter rather than prompt-injecting advice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - Hermes normally depends on PyYAML
    try:
        # Current Hermes ships no PyYAML; hermes_yaml exposes the same
        # safe_load/safe_dump surface (ruamel-backed).
        import hermes_yaml as yaml  # type: ignore[no-redef]
    except Exception:
        yaml = None

logger = logging.getLogger(__name__)

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"ultra": "max"}

# Per-model wire ladders (mirrors hermes agent/reasoning_effort.py). Used only when
# Hermes' own resolver is unavailable. GPT-6 Sol/Luna (and gpt-5.6) accept max;
# Terra stops at xhigh; Astra has no disable/minimal level.
_GPT6_MAX_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_GPT6_TERRA_EFFORTS = ("none", "low", "medium", "high", "xhigh")
_GPT6_ASTRA_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_CODEX_LEGACY_EFFORTS = ("none", "low", "medium", "high", "xhigh")
_CLAUDE_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
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
    "semantic_classifier_model": "gpt-6-luna",
    "semantic_classifier_api_key": "",
    "semantic_classifier_timeout_seconds": 8,
    "semantic_classifier_min_confidence": 0.75,
    "semantic_classifier_max_chars": 1200,
    # Carry task complexity across terse approvals like "yes" / "go ahead".
    "pending_intent_enabled": True,
    "pending_intent_ttl_minutes": 30,
    # Clamp each route onto the session model's real effort ladder
    # (GPT-6 Terra has no max, Astra has no none, ...).
    "model_aware_clamp": True,
    # Leave a manual /reasoning session override alone until /reasoning reset.
    "respect_manual_override": True,
    # Short messages carrying a URL/path are usually "go research this" and
    # should not take the quick/simple low route.
    "url_floor": "medium",
    # Session momentum: a terse follow-up ("continue", "still broken", "same for
    # the other one") right after a tool-heavy turn is part of that task, not
    # small talk. Tool calls are counted from the turn Hermes just finished.
    "momentum_enabled": True,
    "momentum_ttl_minutes": 45,
    "momentum_min_tool_calls": 5,
    "momentum_heavy_tool_calls": 15,
    # Effort floors for short messages that the length heuristic used to send to
    # low: reports of something broken, research/compare asks, and status
    # questions about a live system.
    "troubleshoot_floor": "medium",
    "research_floor": "medium",
    "live_lookup_floor": "medium",
    # Extra nouns (host names, app names) that mark a message as being about a
    # live system. Merged with the built-in generic list.
    "live_system_terms": [],
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
    r"\b(be\s+thorough|flesh\s+out|boil\s+the\s+ocean|do\s+the\s+whole\s+thing|end\s+to\s+end)\b",
    r"\b(architecture|architectural|design\s+decision|tradeoff|migration\s+plan)\b",
    r"\b(?:production\s+(?:system|service|deploy|deployment)|prod\s+(?:deploy|deployment)|rollback\s+safety|rollback-safe|data\s+loss)\b",
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

# Nouns that only mean risk in a technical context ("secret santa", "security
# deposit", "the incident at the zoo"). See _xhigh_keyword_route.
_XHIGH_SECURITY_NOUN_RE = re.compile(r"\b(?:security|auth|oauth|credentials?|secrets?|permissions?|tokens?|ssrf|injection)\b", re.I)
_XHIGH_EVENT_NOUN_RE = re.compile(
    r"\b(?:production\s+(?:incident|outage)|prod\s+(?:incident|outage)|incident|outage|strategy)\b", re.I
)
_SECURITY_CHANGE_VERB_RE = re.compile(
    r"\b(?:rotat\w*|revok\w*|reset|change|add|remove|delete|store|stor(?:e|ing)|leak(?:ed|ing)?|expos\w*|share|"
    r"commit|push|move|migrat\w*|set\s*up|setup|configur\w*|implement|fix|updat\w*|grant|give|audit|harden|review|"
    r"scope|enable|disable|lock|protect|encrypt|generate|create|replace|renew|handle|refresh|wire|hook|integrate)\b",
    re.I,
)
_TECH_CONTEXT_RE = re.compile(
    r"\b(?:api|app|apps|code|repo|server|login|sso|jwt|ssh|key|keys|env|config|plugin|bot|oauth|db|user|users|"
    r"account|accounts|endpoint|cookie|session|scope|scopes|header|headers|webhook|cli|sdk|infra|cluster|backend|"
    r"frontend|deploy|deployment|release|branch|pipeline|ci|prod|production|staging|service|services|network|"
    r"firewall|vpn|cloud|aws|gcp|azure|github|docker|linux|mac|macos|windows|vault|1password|bitwarden|caching|cache|"
    r"database|data|backup|backups|migration|rollout|testing|tests|postmortem|post-mortem|rca)\b",
    re.I,
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
        # "status update" / "progress update" is a status check, not a migration.
        r"\b(?<!status )(?<!progress )(update|upgrade|migrate|migration|schema|database|rollback|backup|restore)\b",
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
# session_key -> reasoning config this router last wrote. Anything else found in the
# session override slot came from a human /reasoning and is left alone.
_ROUTER_SET: dict[str, dict[str, Any]] = {}
# Per-session record of the last finished turn (tool calls, time). Lets a terse
# follow-up inherit the weight of the task it continues.
_MOMENTUM: dict[str, dict[str, Any]] = {}
# Set when a config file exists but cannot be parsed. Routing from defaults in
# that state would silently ignore the user's clamps and platform list. Kept per
# thread: the agent thread re-reads config while the event loop routes.
_CONFIG_STATE = threading.local()
_MOMENTUM_LOCK = threading.Lock()


def _config_error() -> str | None:
    return getattr(_CONFIG_STATE, "error", None)


def _set_config_error(value: str | None) -> None:
    _CONFIG_STATE.error = value
_LAST_HEALTH: dict[str, dict[str, Any] | None] = {
    "route": None,
    "override": None,
    "decision_log": None,
}


def _sweep_stale_state(config: dict[str, Any]) -> None:
    """Drop expired pending intents and momentum so rolled session ids don't pile up."""
    try:
        for sid in list(_PENDING_INTENTS):
            _active_pending_intent(sid, config)  # pops expired/consumed entries
        ttl = _momentum_ttl(config)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=ttl)
        for sid, state in list(_MOMENTUM.items()):
            at = state.get("at") if isinstance(state, dict) else None
            if not isinstance(at, datetime) or at < cutoff:
                _MOMENTUM.pop(sid, None)
    except Exception:
        logger.debug("reasoning-router: state sweep failed", exc_info=True)


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
    # Prefer the profile-scoped check the gateway itself uses; fall back to the
    # plain one on older hosts.
    auth_fn = getattr(gateway, "_is_user_authorized_for_source", None)
    if not callable(auth_fn):
        auth_fn = getattr(gateway, "_is_user_authorized", None)
    if callable(auth_fn):
        try:
            if not auth_fn(getattr(event, "source", None)):
                return None
        except Exception:
            logger.debug("reasoning-router: authorization check failed", exc_info=True)
            return None

    text = _strip_gateway_wrappers(str(getattr(event, "text", "") or ""))
    if not text.strip():
        return None

    # Built-in/plugin slash commands should keep their own semantics. In
    # particular, /reasoning must be able to set manual state without us racing
    # it from the pre-dispatch hook.
    _install_manual_write_hook(gateway)
    if _looks_like_slash_command(text):
        if _is_reasoning_slash_command(text) and _reasoning_command_changes_override(text):
            # Whatever a typed /reasoning sets next is the human's, even when it
            # equals the router's pick. When the setter hook is installed the
            # successful write itself also records ownership, which covers
            # picker callbacks that never pass through this pre-dispatch hook.
            key = _session_key_for(event, gateway, session_store)
            if key:
                _router_set_map(gateway).pop(key, None)
        if not _slash_command_preserves_pending(text):
            _consume_pending_intent_for_event(event, gateway, session_store)
        return None

    config = _router_config(gateway)
    config_error = _config_error()
    if config_error:
        logger.debug("reasoning-router: not routing; %s", config_error)
        _record_health("config", status="unreadable", error=config_error)
        # Refusing to route must not leave the last pick steering the session.
        _release_router_override(gateway, _session_key_for(event, gateway, session_store))
        return None
    if not _truthy(config.get("enabled", True)):
        # Turning the router off must not leave its last pick (possibly
        # low/none) pinned on the session.
        _release_router_override(gateway, _session_key_for(event, gateway, session_store))
        return None

    if not _platform_enabled(event, config):
        logger.debug("reasoning-router: platform not enabled; allowing without override")
        _release_router_override(gateway, _session_key_for(event, gateway, session_store))
        return None

    session_key = _session_key_for(event, gateway, session_store)
    if not session_key:
        logger.debug("reasoning-router: no session key; allowing without override")
        return None

    if _truthy(config.get("respect_manual_override", True)) and _has_manual_override(gateway, session_key):
        logger.debug("reasoning-router: manual /reasoning override active; leaving it")
        _record_health("override", status="manual", session_key=session_key)
        return None

    effort, reason, pending_intent = _effective_effort_for_message(
        text,
        config,
        session_store=session_store,
        session_key=session_key,
    )
    if _truthy(config.get("model_aware_clamp", True)):
        model = _session_model(gateway, session_key)
        fitted = _clamp_to_model(effort, model)
        if fitted != effort:
            reason = f"{reason}; clamped {effort}->{fitted} for {model}"
            effort = fitted
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
        # Shadow mode logs only; a pick left over from live mode would keep
        # steering the model while the log claims nothing was applied.
        _release_router_override(gateway, session_key)
        _record_health(
            "override",
            status="shadow",
            session_key=session_key,
            effort=effort,
        )
    else:
        try:
            _set_reasoning_override(gateway, session_key, reasoning_config)
            _router_set_map(gateway)[session_key] = dict(reasoning_config)
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
    if _config_error() or not _truthy(config.get("enabled", True)):
        return None

    _sweep_stale_state(config)
    try:
        _record_momentum(sid, conversation_history, config)
    except Exception:
        logger.debug("reasoning-router: momentum record failed", exc_info=True)

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

    if command not in {"help", "?", "test", "recent"}:
        _read_router_config_from_disk(include_runtime_override=False)
        config_error = _config_error()
        if config_error:
            return (
                f"Reasoning router config is unreadable, so nothing was changed: {config_error}\n"
                "Fix or remove the file, then retry."
            )

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


_CLASSIFY_HEAD_CHARS = 4000
_CLASSIFY_TAIL_CHARS = 2000


def classify_message(text: str, config: dict[str, Any] | None = None) -> tuple[str, str]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    text = str(text or "")
    # Only the ask matters for routing. A 50k-char paste is read as its opening
    # plus its end, which keeps every pattern bounded in time.
    if len(text) > _CLASSIFY_HEAD_CHARS + _CLASSIFY_TAIL_CHARS:
        text = text[:_CLASSIFY_HEAD_CHARS] + "\n" + text[-_CLASSIFY_TAIL_CHARS:]
    normalized = " ".join(text.strip().split())
    lowered = _normalize_for_match(normalized)
    # Keyword checks run on a masked copy so "tokens per second" or
    # "permission denied" do not read as credential or access-control work,
    # and "don't wipe the drive, just show smart status" is not a wipe.
    risk_lowered = _risk_view(_strip_negated_clauses(lowered))
    effort, reason = _classify_core(text, normalized, lowered, risk_lowered, cfg)
    return effort, reason


def _classify_core(
    text: str,
    normalized: str,
    lowered: str,
    risk_lowered: str,
    cfg: dict[str, Any],
) -> tuple[str, str]:
    if _EFFORT_WORD_RE.search(lowered) and not _is_simple_factual_question(lowered):
        # Several directives in one message resolve together; the strongest
        # wins, then config and model limits apply ("use maximum reasoning and
        # think hard" is a max request, not xhigh).
        if _matches(_COMPILED_MAX, lowered) or _MAX_DIRECTIVE_ANY_RE.search(lowered):
            return _clamp_effort("max", cfg), "explicit maximum/Ultra reasoning request"
        return _clamp_effort("xhigh", cfg), "explicit request for extra-high reasoning"

    # Thanks, venting and sarcasm can quote risky words without asking for
    # anything: "thanks that migration went fine", "rotate all keys lol as if".
    if _is_social_aside(lowered):
        if _matches(_COMPILED_NONE, lowered):
            return _clamp_effort("none", cfg), "matched no-op/simple time-date request"
        return _clamp_effort("low", cfg), "thanks/aside with no new task"

    if any(pattern.search(lowered) for pattern in _COMPILED_PERSONAL_QUICK):
        return _clamp_effort("low", cfg), "personal quick action"

    if _matches(_COMPILED_XHIGH_EXTRA, risk_lowered) and not _is_simple_factual_question(lowered):
        if _is_pure_question(lowered) and not _is_imperative_request(lowered):
            return _clamp_effort("high", cfg), "question about a fleet-wide/irreversible/production change"
        return _clamp_effort("xhigh", cfg), "fleet-wide, irreversible, credential or production change"

    return _classify_legacy(text, normalized, lowered, risk_lowered, cfg)


def _classify_legacy(
    text: str,
    normalized: str,
    lowered: str,
    risk_lowered: str,
    cfg: dict[str, Any],
) -> tuple[str, str]:

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
    xhigh_keywords = _matches(_COMPILED_XHIGH, risk_lowered) or _risk_noun_in_context(risk_lowered)
    if xhigh_keywords:
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
        if _is_pure_question(lowered) and not _is_imperative_request(lowered):
            # Asking how something risky works is not doing it.
            return _clamp_effort("high", cfg), "question about a risky or sensitive topic"
        return _clamp_effort("xhigh", cfg), "matched xhigh complexity/risk keywords"

    if _matches(_COMPILED_IMPLEMENTATION_APPROVAL, risk_lowered):
        return _clamp_effort("high", cfg), "matched implementation approval/tweak request"

    high_groups = _matched_high_groups(risk_lowered)
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

    if high_groups and _is_short_status_question(lowered, cfg):
        return _clamp_effort("medium", cfg), f"question touching {', '.join(high_groups)}"

    topic_only = _single_topic_route(lowered, high_groups, cfg)
    if topic_only is not None:
        return topic_only

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

    if _URL_OR_PATH_RE.search(normalized) and not _matches(_COMPILED_LOW, lowered):
        floor = _normalize_effort_name(cfg.get("url_floor") or "")
        if floor in EFFORT_ORDER:
            return _clamp_effort(floor, cfg), "link/path to look into"

    short_route = _route_short_or_plain(normalized, lowered, cfg)
    if short_route is not None:
        return short_route

    if _matches(_COMPILED_MEDIUM, lowered):
        return _clamp_effort("medium", cfg), "matched normal tool/status keywords"

    default = _normalize_effort_name(cfg.get("default") or DEFAULT_CONFIG["default"])
    if default not in EFFORT_ORDER:
        default = DEFAULT_CONFIG["default"]
    return _clamp_effort(default, cfg), "default route"


def _route_short_or_plain(normalized: str, lowered: str, cfg: dict[str, Any]) -> tuple[str, str] | None:
    """Decide messages no keyword category claimed.

    Signals checked before length: troubleshooting, research/compare, live
    status lookups, and short imperatives. Length only decides when none fire.
    """
    if _is_closer(lowered):
        return _clamp_effort("low", cfg), "quick/simple message"

    troubleshooting = _matches(_COMPILED_TROUBLESHOOT, lowered)
    research = _matches(_COMPILED_RESEARCH, lowered)
    if troubleshooting and (research or _DESIGN_WORD_RE.search(lowered)):
        return _clamp_effort("high", cfg), "troubleshooting that needs research or a design call"
    if troubleshooting:
        floored = _floor_effort("low", "troubleshoot_floor", cfg)
        if floored:
            return floored, "something is broken or misbehaving"
    if research:
        floored = _floor_effort("low", "research_floor", cfg)
        if floored:
            return floored, "research/compare/recommendation request"
    if _DESIGN_WORD_RE.search(lowered):
        return _clamp_effort("high", cfg), "architecture/design tradeoff"
    if _is_live_lookup(lowered, cfg):
        floored = _floor_effort("low", "live_lookup_floor", cfg)
        if floored:
            return floored, "status lookup on a live system"
    if _MULTI_GO_RE.search(lowered):
        return _clamp_effort("high", cfg), "go-ahead for several steps"
    if _BUILD_ARTIFACT_RE.search(_strip_request_prefix(lowered)) and not _is_pure_question(lowered):
        return _clamp_effort("high", cfg), "request to write or build something new"
    if _is_imperative_request(lowered) and (_mentions_live_system(lowered, cfg) or len(normalized) > 40):
        return _clamp_effort("medium", cfg), "short request to change or build something"
    if _OPTION_PICK_RE.match(lowered):
        return _clamp_effort("medium", cfg), "picked one of the offered options"
    if _BARE_CONTINUE_RE.match(lowered):
        return _clamp_effort("medium", cfg), "continue the current task"

    if _matches(_COMPILED_LOW, lowered) or len(normalized) <= _safe_int(cfg.get("low_char_limit"), 80):
        return _clamp_effort("low", cfg), "quick/simple message"
    return None


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
    effort, reason = _apply_momentum(text, effort, reason, session_id, config)
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


def _turn_tool_calls(conversation_history) -> int:
    """Count tool calls made after the last user message in the finished turn."""
    if not isinstance(conversation_history, (list, tuple)):
        return 0
    count = 0
    for message in reversed(conversation_history):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            break
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, (list, tuple)):
                count += len(calls)
    return count


def _record_momentum(session_id: str, conversation_history, config: dict[str, Any]) -> None:
    if not _truthy(config.get("momentum_enabled", True)):
        _MOMENTUM.pop(session_id, None)
        return
    entry = {"tool_calls": _turn_tool_calls(conversation_history), "at": datetime.now(timezone.utc)}
    with _MOMENTUM_LOCK:
        _MOMENTUM[session_id] = entry
        if len(_MOMENTUM) > 512:
            items = sorted(_MOMENTUM.items(), key=lambda item: item[1].get("at") or datetime.min.replace(tzinfo=timezone.utc))
            for key, _state in items[: len(_MOMENTUM) - 512]:
                _MOMENTUM.pop(key, None)


_CONTINUE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|sure|cool|great|nice|perfect|alright)[,.!\s]+)?(?:please\s+)?"
    r"(?:continue|keep\s+going|go\s+on|carry\s+on|proceed|go\s+ahead|do\s+it|finish(?:\s+it)?|resume|"
    r"try\s+again|retry|again|next|and\s+now|same|still|also|what\s+about|now\s+(?:do|try|check|fix))\b",
    re.I,
)


_ANAPHORA_RE = re.compile(
    r"\b(?:it|that|this|them|those|these|the\s+(?:other|rest|next|last|same)|others|same|again|too|as\s+well|"
    r"instead|there|one\s+more|another|remaining|rest)\b",
    re.I,
)


def _momentum_ttl(config: dict[str, Any]) -> int:
    ttl = _safe_int(config.get("momentum_ttl_minutes"), DEFAULT_CONFIG["momentum_ttl_minutes"])
    return min(max(1, ttl), 10_080)


_FOLLOWUP_LEADS = frozenset({"and", "so", "but", "then", "also", "or", "plus", "why", "wait", "what's", "whats", "how's", "hows", "did", "does", "is", "was", "can", "could", "should", "will"})
_GREETING_RE = re.compile(
    r"^(?:good\s+(?:morning|afternoon|evening|night)|morning|gm|hi|hello|hey|yo|sup|howdy)\b[\s!.,]*(?:there|all|team)?[\s!.]*$",
    re.I,
)


def _is_new_standalone_topic(text: str, lowered: str) -> bool:
    """A greeting or self-contained general question that does not point back at the task."""
    if _CONTINUE_RE.match(lowered) or _is_affirmative(text) or _OPTION_PICK_RE.match(lowered):
        return False
    if _ANAPHORA_RE.search(lowered):
        return False
    if _GREETING_RE.match(lowered):
        return True
    if not _is_pure_question(lowered):
        return False
    words = re.findall(r"[a-z0-9']+", lowered)
    if len(words) < 4 or words[0] in _FOLLOWUP_LEADS:
        return False  # "and then?", "why?", "so what now?" lean on the last turn
    # A question about the systems being worked on is still part of the task.
    return not (_mentions_live_system(lowered, None) or _matched_high_groups(_risk_view(lowered)))


def _apply_momentum(
    text: str,
    effort: str,
    reason: str,
    session_id: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Raise short follow-ups that continue a tool-heavy turn.

    Only lifts efforts below the floor and never touches closers ("thanks",
    "lol") or rejections, so a finished task does not keep costing effort.
    """
    if not session_id or not _truthy(config.get("momentum_enabled", True)):
        return effort, reason
    state = _MOMENTUM.get(session_id)
    if not state:
        return effort, reason
    ttl = _momentum_ttl(config)
    if datetime.now(timezone.utc) - state["at"] > timedelta(minutes=ttl):
        _MOMENTUM.pop(session_id, None)
        return effort, reason
    tools = _safe_int(state.get("tool_calls"), 0)
    min_tools = _safe_int(config.get("momentum_min_tool_calls"), DEFAULT_CONFIG["momentum_min_tool_calls"])
    if tools < max(1, min_tools):
        return effort, reason
    lowered = _normalize_for_match(text)
    if _is_closer(lowered) or _is_rejection(text):
        return effort, reason
    if EFFORT_ORDER.index(effort) >= EFFORT_ORDER.index("high"):
        return effort, reason
    if _is_new_standalone_topic(text, lowered):
        # "what is the capital of france" after a big task is a new topic.
        return effort, reason
    heavy = _safe_int(config.get("momentum_heavy_tool_calls"), DEFAULT_CONFIG["momentum_heavy_tool_calls"])
    floor = "high" if tools >= heavy and (_CONTINUE_RE.match(lowered) or _is_affirmative(text)) else "medium"
    lifted = _max_effort((effort, floor), config)
    if lifted == effort:
        return effort, reason
    return lifted, f"follow-up to a {tools}-tool-call turn ({reason})"


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


_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[?.!;:])\s+|,\s*|\s+(?:and\s+)?(?:then|also|afterwards?|after\s+that|next)\s+"
    r"|\s+(?:and|but|plus|however)\s+(?:please\s+)?(?=[a-z])"
)
_WORK_WORD_RE = re.compile(
    r"\b(?:fix|debug|implement|build|change|modify|configure|deploy|restart|reboot|delete|remove|migrate|patch|"
    r"update|upgrade|review|audit|secure|rotate|revoke|wipe|drop|purge|prune|kill|reset|install|write|create|"
    r"set\s*up|clean\s*up|push|merge|rollback|roll\s+back|destroy|truncate|expose|run)\b"
)


def _later_clause_has_work(lowered: str) -> bool:
    """True when any clause after the first asks for work.

    "What does OAuth mean? Rotate the API keys" is a definition plus a task;
    cheap informational shortcuts only apply when the whole message is a question.
    """
    clauses = [c.strip(" .!?") for c in _CLAUSE_SPLIT_RE.split(str(lowered or "")) if c and c.strip(" .!?")]
    for clause in clauses[1:]:
        core = _strip_request_prefix(clause)
        if _IMPERATIVE_VERB_RE.match(core) or _SOCIAL_REQUEST_RE.match(clause):
            return True
        if not _QUESTION_START_RE.match(clause) and _WORK_WORD_RE.search(clause):
            return True
    return False


def _is_simple_factual_question(lowered: str) -> bool:
    text = str(lowered or "").strip()
    if not text:
        return False
    if _later_clause_has_work(text):
        return False
    if re.match(r"^what\s+does\s+[\w .'/-]{1,40}\s+do[?!.\s]*$", text):
        return True
    if re.match(r"^(?:what\s+does|what'?s|whats)\s+[\w .'-]{1,40}\s+(?:mean|stand\s+for)\b", text) or re.match(
        r"^define\s+\S", text
    ):
        return len(text) <= 120
    if not re.match(r"^(?:what|who|when|where)\s+(?:is|are|was|were)\b", text):
        return False
    if re.search(r"\b(?:fix|debug|implement|build|change|modify|configure|deploy|restart|delete|remove|migrate|patch|update|review|audit|secure|rotate)\b", text):
        return False
    # "what are people using for X", "what is wrong with Y", "what is running
    # on the server" need research, diagnosis or a live check, not recall.
    if _matches(_COMPILED_RESEARCH, text) or _matches(_COMPILED_TROUBLESHOOT, text):
        return False
    if _is_live_lookup(text, None):
        return False
    return len(text) <= 120


# ---------------------------------------------------------------------------
# Short-message understanding
#
# The original fallback sent every message under ``low_char_limit`` to low.
# Replaying real chat history showed that short messages are often the most
# expensive ones: "still broken", "continue", "why is plex buffering",
# "rotate the api keys across the fleet". The helpers below recognise those
# shapes so length only decides the route when nothing else does.
# ---------------------------------------------------------------------------

_REQUEST_PREFIX_RE = re.compile(
    r"^(?:(?:can|could|would|will|may)\s+(?:you|u|we|ya)\b(?:\s+please)?[\s,]*"
    r"|(?:ok(?:ay)?|k|so|and|also|now|then|alright|hey|yo|pls|please|plz|kindly|just|"
    r"can|could|would|will|may)\b[\s,]*"
    r"|(?:let'?s|lets|let\s+us)\b[\s,]*"
    r"|i\s+(?:want|need|would\s+like|'?d\s+like|wanna)\s+(?:you\s+|u\s+)?(?:to\s+)?"
    r"|go\s+ahead\s+and\s+|time\s+to\s+|help\s+me\s+|try\s+to\s+|would\s+you\s+mind\s+)+",
    re.I,
)

_IMPERATIVE_VERB_RE = re.compile(
    r"^(?:write|make|add|create|build|set\s*up|setup|schedule|script|rename|de-?dupe|bump|"
    r"redeploy|deploy|install|reinstall|uninstall|configure|reconfigure|enable|disable|change|"
    r"update|upgrade|downgrade|move|migrate|convert|generate|automate|hook\s+up|wire\s+up|"
    r"implement|refactor|patch|fix|repair|register|remove|delete|clean\s*up|cleanup|switch|"
    r"replace|merge|commit|push|restart|reboot|start|stop|revert|roll\s*back|rollback|apply|"
    r"integrate|turn\s+(?:on|off)|rewrite|redo|rebuild|port|backfill|sync|import|export|"
    r"prune|purge|rotate|expose|lock\s+down|harden|optimi[sz]e|tune|speed\s+up|clone|fork|"
    r"release|publish|ship|wire|connect|mount|point|route|forward|allow|block|grant|revoke|"
    r"get\s+(?:the|a|an|me|us)|download|organi[sz]e|dedupe|backup|back\s+up|restore|"
    r"schedule|test|retest|benchmark|debug|troubleshoot|investigate|diagnose|audit|review|code|plan|design|architect|"
    r"walk\s+(?:me\s+)?through|guide\s+(?:me\s+)?through|regenerate|kill|sudo|chown|chmod|nuke|wipe|format|reset|"
    r"decommission|drop|destroy|truncate|flush|open|put|swap)\b",
    re.I,
)

# Personal quick actions that sound like implementation but are one tool call.
_PERSONAL_QUICK_PATTERNS = (
    r"^(?:please\s+)?(?:add|put)\b.{0,60}\bto\s+(?:my|the)\s+(?:grocery|shopping|todo|to-do|to\s+do|packing)\s+list\b",
    r"^(?:please\s+)?remind\s+me\b(?!.{0,80}\bevery\b)",
    r"^(?:please\s+)?(?:text|message|tell|call)\s+(?:my\s+)?(?:wife|husband|mom|dad|partner)\b",
)

_CLOSER_WORDS = frozenset(
    """thanks thank you so much ty thx tysm cheers lol haha hah heh lmao lmfao rofl nice cool
    sweet great perfect awesome amazing wow yay love it that worked works working got
    good all sounds appreciate appreciated legend beautiful brilliant excellent neat
    man dude bro buddy nerd one oh ah ahh ahhh okay ok k yeah yep nice gg gn night goodnight btw fyi
    morning gm hi hey hello sup yo hiii hii there you""".split()
)
_CLOSER_CORE_WORDS = frozenset(
    """thanks thank ty thx tysm cheers lol haha hah heh lmao lmfao rofl nice cool sweet great
    perfect awesome amazing wow yay love worked appreciate appreciated legend beautiful
    brilliant excellent neat gg gn night goodnight morning gm hi hey hello sup yo hiii hii""".split()
)

_TROUBLESHOOT_PATTERNS = (
    r"\b(?:isn'?t|aren'?t|wasn'?t|not|never|won'?t|wont|can'?t|cant|cannot|doesn'?t|didn'?t|didnt|doesnt|isnt|arent|"
    r"don'?t|dont|aint|ain'?t)\s+"
    r"(?:\w+\s+){0,2}(?:work|working|load|loading|respond|responding|show|showing|showin|online|up|playing|play|"
    r"import|importing|resolve|resolving|fire|firing|run|running|start|starting|connect|connecting|sync|syncing|"
    r"send|sending|go\s+off|reachable|picking\s+up|pick\s+up|turn\s+on|boot|booting|update|updating|save|saving|"
    r"open|opening|download|downloading|reply|replying|load)\b",
    r"\b(?:keeps?|kept)\s+(?:on\s+)?(?:restarting|crashing|buffering|climbing|failing|dropping|disconnecting|"
    r"timing\s+out|freezing|hanging|looping|stuttering|rebooting|going\s+down|dying|erroring|growing|increasing|rising|"
    r"filling(?:\s+up)?|beeping|flapping)\b",
    r"\b(?:is|are|went|goes|been|was|still|it'?s|its)\s+down\b|\bdown\s+again\b|\b(?:dead|dying|died|dies|flapping|flaky|"
    r"acting\s+(?:up|weird|funny|strange)|blank|white\s+screen|beeps?|beeping|latency|spikes?|spiking|pinned|pegged|maxed\s+out|"
    r"drops|dropping|expired|disappear(?:s|ing)?|wtf)\b",
    r"\b(?:can'?t|cant|cannot|unable\s+to|couldn'?t|could\s+not)\s+(?:\w+\s+)?(?:ssh|reach|access|log\s*in|login|connect|"
    r"ping|mount|see|find|get\s+(?:in|to|into))\b",
    r"\bshows?\s+(?:no|0|zero|nothing)\b|\bno\s+data\b|\bsays\s+nothing\b|\b(?:it|now\s+it|this|that)\s+says\b|"
    r"\bconnection\s+reset\b|\breset\s+by\s+peer\b|\btimes?\s+out\b|\b(?:cert|certificate|ssl|tls)\s+(?:warning|error)s?\b|"
    r"\bpending\s+sectors?\b|\breallocated\b|\bat\s+(?:9\d|100)\s*%|\bsomething\s+(?:is\s+|'?s\s+)?(?:wrong|off)\b|"
    r"\bsame\s+(?:issue|problem|error)\b|\bstill\s+(?:the\s+)?same\b|\bstill\s+(?:happening|broken|failing|not)\b|"
    r"\bno\s+idea\s+why\b|\bgoing\s+on\s+with\b",
    r"\b(?:works?|resolves?|loads?|connects?)\b.{0,50}\bbut\b.{0,50}\b(?:not|times?\s+out|fails?|doesn'?t|won'?t|can'?t)\b",
    r"\bwhat'?s\s+(?:causing|eating|using|hogging|filling|killing)\b|\bwhats\s+(?:causing|eating|using|hogging|filling)\b|"
    r"\b(?:what|who)\s+is\s+(?:causing|eating|using|hogging|filling|killing)\b",
    r"\b(?:offline|broken|broke|failing|failed|errors?|erroring|stuck|stalls?|stalled|stalling|hangs?|hanging|"
    r"frozen|freezing|crash(?:es|ed|ing)?|buffering|stutter(?:s|ing)?|laggy|lagging|slow|timed?\s*out|timeout|"
    r"disconnected|unreachable|unavailable|refused|denied|500|502|503|504|404|oom|leak(?:ing)?|corrupt(?:ed)?|"
    r"missing|vanished|disappeared|went\s+away|blurred|blurry|glitch(?:y|ing)?|not\s+found|no\s+longer)\b",
    r"\b(?:twice|double[ds]?|duplicat(?:e|es|ed|ing)|every\s+(?:message|time)\s+now|out\s+of\s+sync|wrong\s+"
    r"(?:time|date|language|audio|subtitles?|user|account|order)|says\s+no\b|won'?t\s+stop|keeps?\s+(?:asking|saying|"
    r"showing|sending|posting|replying))\b",
    r"\b(?:why\s+(?:is|are|does|do|did|was|were|isn'?t|won'?t|can'?t|would|has|have|this|it|the|my|our)|"
    r"what'?s\s+(?:wrong|going\s+on|broken|happening|up\s+with)|whats\s+(?:wrong|going\s+on|broken|happening|up\s+with)|"
    r"what\s+(?:is|went)\s+wrong|any\s+idea\s+why|how\s+come|what\s+happened)\b",
)

_RESEARCH_PATTERNS = (
    r"\b(?:compare|comparison|comparing|vs\.?|versus|pros\s+and\s+cons|worth\s+it|is\s+it\s+worth|"
    r"worth\s+(?:switching|moving|upgrading|getting|buying|it)|which\s+(?:\w+\s+){0,2}(?:is|would\s+be)\s+better|"
    r"what'?s\s+better|whats\s+better|what\s+is\s+better|best\s+(?:\w+\s+){0,2}(?:way|option|approach|practice|tools?|setup|"
    r"apps?|nas|router|choice|pick|software|service|provider|distro|os|hardware|gpu|cpu|drive|drives|ssd)|"
    r"(?:what\s+is|what'?s|whats)\s+the\s+best\b|is\s+(?:\w+\s+){0,4}enough\b|enough\s+for\b|is\s+it\s+ok(?:ay)?\s+to\b|"
    r"alternatives?\s+(?:to|for)|options?\s+for|look\s+into|looking\s+into|look\s+up|research|recommend(?:ation)?s?|"
    r"should\s+(?:i|we)\s+(?:use|go|switch|try|move|get|buy|pick|stick|keep|run|choose)|"
    r"difference\s+between|what\s+would\s+it\s+take|opinion|thoughts\s+on|your\s+take|summari[sz]e|"
    r"what\s+changed|how\s+does\b.{0,60}\bcompare|what\s+are\s+(?:people|folks|others)\s+using|"
    r"is\s+it\s+safe|how\s+do\s+(?:i|we)\s+(?:safely\s+)?(?:undo|revert|recover|restore|roll\s*back|migrate|move)\b|is\s+(?:it|that|this)\s+(?:a\s+good\s+idea|smart|better|faster|worth)|"
    r"read\s+(?:the|up\s+on)\b.{0,40}\b(?:docs?|documentation|changelog|release\s+notes)|"
    r"any\s+(?:good\s+)?(?:alternatives?|options?|recommendations?)|how\s+(?:can|could|should|would)\s+(?:we|i)\b|"
    r"what\s+(?:do|should|would)\s+(?:we|i|you)\s+(?:need|use|do|recommend|suggest))\b",
)

_LIVE_SYSTEM_TERMS = (
    "server", "servers", "box", "host", "hosts", "nas", "container", "containers", "docker", "compose",
    "service", "services", "cron", "crons", "backup", "backups", "disk", "disks", "drive", "cpu", "gpu",
    "ram", "memory", "uptime", "port", "ports", "dns", "vpn", "node", "nodes", "queue", "bot", "bots",
    "gateway", "proxy", "database", "db", "repo", "repos", "deploy", "deploys", "logs", "log", "cert",
    "certs", "ssl", "tls", "network", "wifi", "router", "vm", "vms", "pool", "volume", "volumes",
    "cluster", "pod", "pods", "site", "website", "domain", "subdomain", "api", "endpoint", "job", "jobs",
    "pipeline", "build", "ci", "workflow", "tunnel", "firewall", "jellyfin", "plex", "emby", "sonarr",
    "radarr", "lidarr", "prowlarr", "bazarr", "overseerr", "jellyseerr", "qbittorrent", "transmission",
    "sabnzbd", "coolify", "tailscale", "wireguard", "home assistant", "homeassistant", "zigbee",
    "traefik", "caddy", "nginx", "kubernetes", "k8s", "portainer", "proxmox", "unraid", "truenas",
    "grafana", "prometheus", "uptime kuma", "pihole", "pi-hole", "adguard", "postgres", "mysql", "redis",
    "hermes", "discord", "github", "gitlab", "cloudflare", "vercel", "ssh",
)

_LIVE_LOOKUP_SHAPE_PATTERNS = (
    r"^(?:is|are)\s+(?:the\s+|my\s+|our\s+)?[\w.-]+(?:\s+[\w.-]+){0,2}\s+(?:up|down|online|offline|running|working|"
    r"healthy|reachable|back|alive|ok|okay|done|finished|synced|updated|full)\b",
    r"^(?:did|has|have|was|were)\s+.{0,80}\b(?:finish|finished|run|ran|complete|completed|come\s+back|came\s+back|"
    r"go\s+off|went\s+off|fire|fired|succeed|succeeded|work|worked|start|started|deploy|deployed|pass|passed|"
    r"sync|synced|update|updated|import|imported|download|downloaded)\b",
    r"\bhow\s+(?:much|many)\s+(?:\w+\s+){0,3}(?:space|disk|storage|ram|memory|cpu|free|left|used|running|"
    r"movies|shows|series|episodes|containers|services|jobs|crons|items|files|users|errors)\b",
    r"\b(?:uptime|cpu\s+temp|temperature|load\s+average|disk\s+usage|free\s+space|space\s+left|status)\b",
    r"\b(?:what|which)\s+(?:\w+\s+)?(?:containers?|services?|apps?|jobs?|crons?|ports?|version|versions|"
    r"models?|nodes?|hosts?|ips?|ip\s+address)\b",
    r"\b(?:right\s+now|rn|currently|at\s+the\s+moment|atm|anything\s+running)\b",
    r"^(?:show|list|check|tell)\s+(?:me\s+)?",
)

_XHIGH_EXTRA_PATTERNS = (
    # fleet-wide changes
    r"\b(?:every|all(?:\s+(?:the|of\s+the|my|our))?|each)\s+(?:\w+\s+)?(?:server|servers|host|hosts|machine|machines|"
    r"node|nodes|box|boxes|service|services|container|containers|repo|repos|clone|clones|site|sites|vm|vms|device|"
    r"devices|environment|environments)\b.{0,80}\b(?:rotate|reset|delete|remove|wipe|migrate|upgrade|update|restart|reboot|"
    r"move|redeploy|prune|purge|rebuild|lock\s+down|replace|reinstall|cut|force|re-?key|patch)\b",
    r"\b(?:rotate|reset|delete|remove|wipe|migrate|upgrade|restart|reboot|shut\s*down|power\s+off|move|redeploy|prune|purge|"
    r"rebuild|lock\s+down|replace|reinstall|re-?key|re-?image|patch|update)\b.{0,100}\b(?:every|all|each|across)\s+(?:the\s+|of\s+the\s+|my\s+|our\s+)?"
    r"(?:\w+\s+)?(?:fleet|servers?|hosts?|machines?|nodes?|box(?:es)?|services?|containers?|repos?|clones?|sites?|"
    r"vms?|devices?|environments?)\b",
    r"\b(?:across\s+(?:the\s+)?(?:fleet|all|every)|fleet[-\s]?wide|on\s+all\s+(?:the\s+)?hosts|(?:whole|entire)\s+fleet)\b",
    # credential rotation / access changes / exposure
    r"\brotat(?:e|ing|ion)\b.{0,60}\b(?:keys?|secrets?|tokens?|credentials?|passwords?|certs?|certificates?)\b",
    r"\b(?:reset|revoke|expire|invalidate)\b.{0,30}\b(?:all|every(?:one)?'?s?|each)\b.{0,30}\b(?:passwords?|tokens?|keys?|"
    r"sessions?|credentials?)\b",
    r"\bput\b.{0,30}\b(?:api\s+keys?|secrets?|tokens?|passwords?|credentials?|\.env)\b.{0,30}\b(?:repo|git|code|public|commit)\b",
    r"\b(?:pentest|pen[-\s]?test|penetration\s+test|vulnerabilit(?:y|ies)|hacked|compromised|breach(?:ed)?|intrusion|"
    r"malware|ransomware|rootkit)\b",
    r"\b(?:give|grant)\b.{0,60}\b(?:write|admin|root|sudo|owner)\s+(?:access|permissions?|rights)\b",
    r"\b(?:expos\w*|open(?:ing)?(?:\s+up)?|make\s+public|put|publish|forward\w*|allow)\b.{0,60}\b(?:(?:open|public)\s+internet|"
    r"the\s+internet|to\s+the\s+world|from\s+anywhere|publicly|wan|outside\s+world)\b",
    r"\b0\.0\.0\.0/0\b|::/0|\bport[-\s]?forward\w*\b",
    r"\bmake\s+(?:the\s+|my\s+|this\s+|our\s+)?(?:\w+\s+)?(?:public|world[-\s]readable)\b",
    # irreversible operations
    r"\b(?:wipe|reformat|re-?format|format\s+the|factory\s+reset|rm\s+-rf|force[-\s]?push|drop\s+(?:the\s+)?"
    r"(?:\w+\s+){0,2}(?:table|tables|database|db|schema)|truncate\s+(?:the\s+)?(?:\w+\s+)?table|rewrite\s+(?:the\s+)?"
    r"(?:git\s+)?history)\b",
    r"\b(?:delete|remove|purge|prune|wipe|nuke)\b.{0,40}\b(?:all|every|entire|whole)\b",
    r"\bgit\s+push\b.{0,40}(?:\s-f\b|--force)|\b(?:terraform|tofu|pulumi)\s+destroy\b|\b(?:zfs|zpool)\s+destroy\b|"
    r"\bmkfs(?:\.\w+)?\b|\bdd\s+if=|\bdocker\s+(?:system|volume|image)\s+prune\b|\bkubectl\s+delete\b|\bhelm\s+uninstall\b|"
    r"\bdecommission\w*\b|\bchmod\s+(?:-r\s+)?777\b|\bch(?:own|mod)\s+-r\b|\bnuke\b|\bmove\s+(?:everything|all\s+of\s+it)\b.{0,60}\b(?:off|shut)\b",
    r"\b(?:swap|point|switch|change|move|update|cut\s+over)\b.{0,40}\b(?:dns|domain|nameservers?|a\s+records?|mx\s+records?)\b",
    # production changes
    r"\b(?:prod|production)\b.{0,60}\b(?:upgrade|migrate|migration|push|deploy|ship|release|rebuild|delete|drop|restart|change|"
    r"update|write|access|reset|move)\b",
    r"\b(?:upgrade|migrate|migration|push|deploy|ship|release|roll\s*out|rebuild|delete|drop|restart|change|update|reset|move|"
    r"uninstall)\b.{0,80}"
    r"\b(?:prod|production)\b",
    # continuity demands
    r"\b(?:zero[-\s]?downtime|cut\s*over|cutover|without\s+(?:any\s+)?(?:downtime|data\s+loss|losing|locking)|"
    r"(?:dont|don'?t|do\s+not|without)\s+(?:lose|losing|lock(?:ing)?\s+(?:me|us)\s+out)|major\s+version|from\s+scratch)\b",
)

# Phrases that trip risk keywords without describing a risky action.
_RISK_MASKS = (
    (re.compile(r"\btokens?\s+(?:per\s+second|/s|per\s+sec|used|usage|count|counts|budget|limit|limits|in\b|out\b|spent|burned)"
                r"|\b(?:how\s+many|input|output|total|context|prompt|completion|cached)\s+tokens?\b", re.I), "units"),
    (re.compile(r"\bpermission\s+denied\b", re.I), "access error"),
    (re.compile(r"\b(?:command|button|script|endpoint|shortcut|way|option|toggle)\s+(?:to|that|for)\s+"
                r"(?:restart|stop|shut\s*down|reboot)", re.I), "control action"),
    (re.compile(r"\b(?:my|our|the|this|your|current|whole|homelab|home)\s+setup\b", re.I), "the rig"),
    (re.compile(r"\bsecret\s+(?:santa|life|garden|recipe|menu|ingredient|passage|level)s?\b"
                r"|\bsecurity\s+(?:deposit|guard|blanket|question)s?\b|\bpermission\s+(?:from|slip)\b"
                r"|\bend[-\s]to[-\s]end\s+encrypt\w*", re.I), "everyday phrase"),
    (re.compile(r"\brm\s+-r?f?r?\s+(?:\./)?(?:node_modules|dist|build|target|out|\.next|\.cache|__pycache__|\.venv|venv|"
                r"/tmp/\S*|tmp|\.pytest_cache|coverage)/?(?=\s|$)", re.I), "clean build output"),
)

_DESIGN_WORD_RE = re.compile(r"\b(?:architecture|architectural|design\s+decision|tradeoffs?|trade-offs?|strategy)\b", re.I)
_EFFORT_WORD_RE = re.compile(
    r"\bxhigh\b|\bthink\s+(?:really\s+)?hard(?:er)?\b"
    r"|\b(?:use|using|with|at|on|go|try|need|want|in|switch\s+to)\s+(?:an?\s+)?extra[\s-]*high\b(?!\s+(?:setting|mode\s+on\s+my|speed|spin|heat))"
    r"|\bextra[\s-]*high\s+(?:reasoning|effort|thinking)\b",
    re.I,
)
# Unanchored max directive, consulted only when another effort directive is in
# the same message so the strongest one can win.
_MAX_DIRECTIVE_ANY_RE = re.compile(
    r"\b(?:use|using|with|at|on|set\s+(?:it\s+)?to|switch\s+to|go)\s+(?:the\s+)?(?:max(?:imum)?|ultra)"
    r"(?:\s+(?:reasoning|effort|thinking))?\b",
    re.I,
)
_QUESTION_START_RE = re.compile(
    r"^(?:explain|describe|what|what'?s|whats|why|how|which|when|where|who|is|are|was|were|does|do|did|should|would|could|"
    r"can\s+(?:i|it|this|that|they)|will\s+(?:it|this|that)|has|have|any|anything)\b",
    re.I,
)
_REQUEST_QUESTION_RE = re.compile(
    r"^(?:(?:can|could|would|will|may)\s+(?:you|u|we|ya)|let'?s|lets|please)\b",
    re.I,
)

_COMPILED_TROUBLESHOOT = tuple(re.compile(p, re.I) for p in _TROUBLESHOOT_PATTERNS)
_COMPILED_RESEARCH = tuple(re.compile(p, re.I) for p in _RESEARCH_PATTERNS)
_COMPILED_LIVE_SHAPE = tuple(re.compile(p, re.I) for p in _LIVE_LOOKUP_SHAPE_PATTERNS)
_COMPILED_XHIGH_EXTRA = tuple(re.compile(p, re.I) for p in _XHIGH_EXTRA_PATTERNS)
_COMPILED_PERSONAL_QUICK = tuple(re.compile(p, re.I) for p in _PERSONAL_QUICK_PATTERNS)


def _normalize_for_match(text: str) -> str:
    """Lowercase, collapse whitespace, and fold typographic quotes (phones send ’)."""
    folded = str(text or "").translate({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"', 0x2014: " ", 0x2013: " "})
    return " ".join(folded.strip().split()).lower()


_QUOTED_SPAN_RE = re.compile(r"```.*?(?:```|$)|`[^`]{1,400}`|\"[^\"]{1,400}\"", re.S)
# Asking to run or apply the quoted text keeps it live: "run `terraform destroy`".
_EXEC_REQUEST_RE = re.compile(
    r"\b(?:run|rerun|re-run|execute|exec|invoke|apply|paste\s+(?:and|&)\s+run|go\s+ahead)\b"
    r"|\b(?:do|try)\s+(?:it|this|that)\b",
    re.I,
)


# A quote that is the object of an imperative action ("reset all user `passwords`",
# "revoke all `tokens`") is an operand, not evidence, so it stays live.
_ACTION_OPERAND_RE = re.compile(
    r"(?:^|[.!?;:\n]\s*|\b(?:please|pls|now|then|and|also|just|can\s+you|could\s+you|go\s+ahead\s+and)\s+)"
    r"(?:reset|revoke|rotate|regenerate|delete|remove|drop|wipe|purge|destroy|truncate|disable|kill|stop|"
    r"restart|reboot|deploy|redeploy|push|force[-\s]?push|migrate|revert|roll\s*back|rollback|chmod|chown|"
    r"expose|grant|change|update|upgrade|downgrade|overwrite|replace|move|rename|unset|clear|flush|expire|"
    r"invalidate|ban|block|unblock|lock|unlock|terminate|uninstall|install|enable|shut\s*down)\b"
    r"(?:\s+[\w'./-]+){0,5}\s*$",
    re.I,
)


def _mask_quoted_evidence(lowered: str) -> str:
    """Quoted/fenced text is evidence (logs, output) unless the user asks to run
    it or it is the operand of an explicit action request."""
    if not any(ch in lowered for ch in "`\""):
        return lowered
    outside = _QUOTED_SPAN_RE.sub(" ", lowered)
    if _EXEC_REQUEST_RE.search(outside):
        return lowered

    def _mask(match: "re.Match[str]") -> str:
        before = _QUOTED_SPAN_RE.sub(" ", lowered[: match.start()])
        if _ACTION_OPERAND_RE.search(before):
            return match.group(0)
        return " quoted text "

    return _QUOTED_SPAN_RE.sub(_mask, lowered)


def _risk_view(lowered: str) -> str:
    view = _mask_quoted_evidence(lowered)
    for pattern, replacement in _RISK_MASKS:
        view = pattern.sub(replacement, view)
    return view


_NEGATED_CLAUSE_RE = re.compile(
    r"\b(?:(?:please\s+)?(?:don'?t|dont|do\s+not)|never|no\s+need\s+to|(?:i\s+am|i'?m|im)\s+not\s+asking\s+(?:you\s+|u\s+)?to|"
    r"not\s+asking\s+(?:you\s+|u\s+)?to|without|skip(?:ping)?|won'?t\s+need\s+to)\s+(?:\w+\s+){0,2}?"
    r"(?:wip(?:e|ing)|drop(?:ping)?|delet(?:e|ing)|remov(?:e|ing)|restart(?:ing)?|reboot(?:ing)?|expos(?:e|ing)|touch(?:ing)?|"
    r"force[-\s]?push(?:ing)?|push(?:ing)?|rotat\w*|the\s+rotation|deploy(?:ing)?|chang(?:e|ing)|modify(?:ing)?|kill(?:ing)?|"
    r"reset(?:ting)?|nuk(?:e|ing)|format(?:ting)?|anything)\b"
    # Mask only the negated clause: stop at punctuation or at a conjunction
    # that starts a new, positive clause ("... but rotate the keys").
    r"(?:(?!\b(?:but|and|then|however|though|although|instead|yet|except|also|plus|so)\b)[^,;.!?:])*",
    re.I,
)

_SOCIAL_START_RE = re.compile(
    r"^(?:thanks|thank\s+you|thx|ty|tysm|cheers|ha(?:ha)+|lol|lmao|lmfao|wow|oh\s+(?:great|nice|wow|man)|fyi|"
    r"just\s+(?:wanna|want\s+to|gonna|need\s+to)\s+vent|just\s+venting|sure,)\b[\s,.!:-]*",
    re.I,
)
_SOCIAL_END_RE = re.compile(
    r"(?:\b(?:but|so)\s+nah|\bnah|\bas\s+if|\blove\s+that\s+for\s+me|\bthanks|\bthank\s+you|\bty|\blol|\blmao|"
    r"\bha(?:ha)+)[\s.!?\U0001F300-\U0001FAFF]*$|\U0001F389\s*$",
    re.I,
)
_SOCIAL_REQUEST_RE = re.compile(
    r"\b(?:can\s+(?:you|u)|could\s+(?:you|u)|would\s+(?:you|u)|will\s+(?:you|u)|please|pls|plz|go\s+ahead|let'?s|lets|"
    r"now\s+(?:do|can|go|run|let|try|fix|check|restart|deploy|push|add)|next\s+(?:do|step|up)|"
    r"also\s+(?:do|can|please|run|fix|check|restart|add|update|deploy|push)|next\s*[:,-]|then\s+(?:can|do|please)|i\s+(?:need|want)\s+(?:you|u)|make\s+sure|do\s+it)\b",
    re.I,
)
_MULTI_GO_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|sure|alright|great|perfect)[,.!\s]+)*(?:please\s+)?(?:do|run|apply|ship|go\s+with|implement)\s+"
    r"(?:all|both|every(?:thing)?|each|them\s+all|all\s+(?:of\s+)?(?:it|them|those|three|four|five|the\s+(?:steps|fixes|changes)))\b"
    r"|^(?:ok(?:ay)?[,\s]+)?do\s+\d+\b.{0,20}\bthen\s+\d+",
    re.I,
)


def _strip_negated_clauses(lowered: str) -> str:
    """Drop "don't wipe the drive" style clauses before risk matching."""
    return _NEGATED_CLAUSE_RE.sub(" ", lowered)


def _is_social_aside(lowered: str) -> bool:
    """Thanks, venting, sarcasm or a status recap that asks for nothing new."""
    text = lowered.strip()
    if not text or len(text) > 300:
        return False
    start = _SOCIAL_START_RE.match(text)
    if not start and not _SOCIAL_END_RE.search(text):
        return False
    if "?" in text or _SOCIAL_REQUEST_RE.search(text):
        return False
    if _matches(_COMPILED_TROUBLESHOOT, _strip_negated_clauses(text)) or _matches(_COMPILED_RESEARCH, text):
        return False
    core = _strip_request_prefix(text[start.end():] if start else text)
    return not _IMPERATIVE_VERB_RE.match(core)


def _risk_noun_in_context(risk_lowered: str) -> bool:
    """Security or incident nouns only count as risk next to a technical verb or object."""
    if _XHIGH_SECURITY_NOUN_RE.search(risk_lowered) and (
        _SECURITY_CHANGE_VERB_RE.search(risk_lowered) or _TECH_CONTEXT_RE.search(risk_lowered)
    ):
        return True
    if _XHIGH_EVENT_NOUN_RE.search(risk_lowered):
        if re.search(r"\b(?:prod|production)\s+(?:incident|outage)\b", risk_lowered):
            return True
        if _TECH_CONTEXT_RE.search(risk_lowered) or _mentions_live_system(risk_lowered, None):
            return True
    return False


def _strip_request_prefix(lowered: str) -> str:
    return _REQUEST_PREFIX_RE.sub("", lowered, count=1).strip()


def _is_pure_question(lowered: str) -> bool:
    """A question asking for an answer, not a polite request to do work."""
    text = lowered.strip()
    if not text or _REQUEST_QUESTION_RE.match(text):
        return False
    # "do not restart X but rotate Y" is an instruction, not a "do ...?" question.
    if re.match(r"(?:do\s+not|don'?t|dont)\b", text):
        return False
    if _later_clause_has_work(text):
        return False
    return bool(_QUESTION_START_RE.match(text)) or text.endswith("?")


def _is_imperative_request(lowered: str) -> bool:
    core = _strip_request_prefix(lowered)
    if not core:
        return False
    match = _IMPERATIVE_VERB_RE.match(core)
    if not match:
        return False
    rest = core[match.end():].strip(" .!?")
    # "set this one please" / "fix" alone carry no task of their own.
    return bool(rest) and not re.fullmatch(r"(?:it|this|that|this\s+one|that\s+one|them|those)(?:\s+please)?", rest)


_BUILD_ARTIFACT_RE = re.compile(
    r"^(?:write|build|create|implement|automate|script|generate|make|code)\b.{0,40}\b(?:script|plist|launchd|cron|"
    r"job|bot|service|tool|app|page|site|dashboard|hook|webhook|plugin|skill|digest|workflow|pipeline|api|cli|"
    r"integration|exporter|importer|parser|scraper|daemon|endpoint|test\s+suite|tests|actions?|program|function|module|"
    r"playbook|dockerfile|unit|alerts?|monitor)\b"
    r"|^script\s+to\b"
    r"|^(?:i\s+)?(?:need|want)\s+(?:a|an)\s+(?:\w+\s+){0,3}(?:script|tool|bot|app|cli|dashboard|service|automation|cron|"
    r"plugin|workflow|action|program|monitor|alert)s?\b",
    re.I,
)
_OPTION_PICK_RE = re.compile(
    r"^(?:(?:option|choice|plan|number|#)\s*)?(?:\d{1,2}|[a-d])[.)!]?(?:\s+(?:please|pls|then|it\s+is))?[.!]?$"
    r"|^(?:the\s+)?(?:first|second|third|last)\s+(?:one|option)\b",
    re.I,
)


_BARE_CONTINUE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|sure|alright|great|perfect)[,.!\s]+)?(?:please\s+)?"
    r"(?:continue|keep\s+going|go\s+on|carry\s+on|proceed|resume|try\s+again|retry|finish\s+(?:it|up|the\s+job))"
    r"(?:\s+please)?[.!]*$",
    re.I,
)


def _is_short_status_question(lowered: str, cfg: dict[str, Any]) -> bool:
    """A short question about state ("when did the backup run") rather than a task."""
    if len(lowered) > 120 or not _is_pure_question(lowered) or _is_imperative_request(lowered):
        return False
    if _later_clause_has_work(lowered):
        return False
    if _DESIGN_WORD_RE.search(lowered) or _matches(_COMPILED_RESEARCH, lowered):
        return False
    return bool(
        re.match(r"^(?:when|what\s+time|did|has|have|was|were|is|are|which|where)\b", lowered)
        or re.match(r"^why\s+did\s+(?:my|the|our|this|it)\b.{0,60}\b(?:not|fail|stop|skip)", lowered)
        or _is_live_lookup(lowered, cfg)
    )


# Groups whose keyword is usually the topic of a message rather than the work
# asked for: "which provider is that on", "what reasoning is sol set to".
_TOPIC_WORD_GROUPS = frozenset({"hermes_internals", "logging_audit", "verification"})
_STATUS_CHECK_RE = re.compile(r"\b(?:status|progress)\s+(?:update|check|report)\b")


def _single_topic_route(lowered: str, high_groups: list[str], cfg: dict[str, Any]) -> tuple[str, str] | None:
    """Route one-category keyword hits that name a topic instead of asking for work.

    Real traffic showed a single high keyword ("update", "reasoning", "log")
    sent quick questions and status checks to high, and those turns mostly
    finished with no tool calls. They go to medium, never below, so a real task
    phrased as a question still gets room to work.
    """
    if _STATUS_CHECK_RE.search(lowered) and len(lowered) <= 120 and not _asks_for_work_anywhere(lowered):
        return _clamp_effort("medium", cfg), "status check"
    if len(high_groups) != 1:
        return None
    group = high_groups[0]
    # Deletions keep the careful route, and "why did X break" questions are
    # investigations: that group had the heaviest real turns of all.
    if group in ("destructive", "debug_forensics"):
        return None
    if _DESIGN_WORD_RE.search(lowered) or _matches(_COMPILED_RESEARCH, lowered):
        return None
    if _matches(_COMPILED_TROUBLESHOOT, lowered) or _asks_for_work_anywhere(lowered):
        return None
    if _is_pure_question(lowered) and len(lowered) <= 200:
        return _clamp_effort("medium", cfg), f"question touching {group}"
    if group in _TOPIC_WORD_GROUPS and not _MULTI_GO_RE.search(lowered) and len(lowered) <= 160:
        return _clamp_effort("medium", cfg), f"mentions {group} without asking for a change"
    return None


_WORK_ASK_RE = re.compile(
    r"\b(?:let'?s|lets|let\s+us|go\s+ahead|yes\s+please|do\s+(?:those|these|that|it|both|all)|"
    r"we\s+(?:need|should|have)\s+to|need\s+to|try\s+(?:some|a|the|it|to))\b"
)


def _asks_for_work_anywhere(lowered: str) -> bool:
    """True when any clause asks for work, even inside a message ending in '?'."""
    if _WORK_ASK_RE.search(lowered):
        return True
    return any(_is_imperative_request(part.strip()) for part in re.split(r"[.!;\n]+|,\s*(?:and\s+)?|\band\s+then\b", lowered) if part.strip())


def _is_closer(lowered: str) -> bool:
    """Thanks / lol / nice: ends a thread rather than continuing its work."""
    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return not re.search(r"\d", lowered)  # emoji/punctuation only; "2" is an option pick
    if re.fullmatch(r"a+y+", words[0] or ""):
        words = words[1:] or ["yay"]
    return all(w in _CLOSER_WORDS for w in words) and any(w in _CLOSER_CORE_WORDS for w in words)


def _live_terms(config: dict[str, Any] | None) -> tuple[str, ...]:
    extra = (config or {}).get("live_system_terms") or []
    if isinstance(extra, str):
        extra = [part.strip() for part in extra.split(",")]
    elif not isinstance(extra, (list, tuple, set, frozenset)):
        extra = []
    cleaned = tuple(str(term).strip().lower() for term in extra if str(term or "").strip())
    return _LIVE_SYSTEM_TERMS + cleaned


def _mentions_live_system(lowered: str, config: dict[str, Any] | None) -> bool:
    for term in _live_terms(config):
        if " " in term or "-" in term:
            if term in lowered:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", lowered):
            return True
    return False


def _is_live_lookup(lowered: str, config: dict[str, Any] | None) -> bool:
    if not _matches(_COMPILED_LIVE_SHAPE, lowered):
        return False
    return _mentions_live_system(lowered, config)


def _floor_effort(effort: str, floor_key: str, cfg: dict[str, Any]) -> str | None:
    floor = _normalize_effort_name(cfg.get(floor_key) or "")
    if floor not in EFFORT_ORDER:
        return None
    return _max_effort((effort, floor), cfg)


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
        or re.search(r"(?m)^[ \t]*[-\w.\"']+[ \t]*:[ \t]*\S", raw)
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


_YAML_CACHE: dict[str, tuple[bytes, dict[str, Any] | str]] = {}


def _read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        _set_config_error(f"no YAML parser available to read {path}")
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _set_config_error(f"failed to read {path}: {exc}")
        return {}
    # Key on content, not mtime: a same-size edit inside one mtime tick would
    # otherwise serve the old config. Hashing a small file costs microseconds;
    # the YAML parse is what we skip.
    stamp = hashlib.blake2b(raw, digest_size=16).digest()
    cached = _YAML_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        if isinstance(cached[1], str):
            # Known-bad file, unchanged: same error, no re-parse, no log spam.
            _set_config_error(cached[1])
            return {}
        return dict(cached[1])
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
        # Only an empty document means "no settings". [], false or 0 are
        # malformed configs and must stay errors, not silently become {}.
        if data is None:
            data = {}
    except Exception as exc:
        logger.warning("reasoning-router: failed to read %s: %s", path, exc)
        error = f"failed to read {path}: {exc}"
        _YAML_CACHE[str(path)] = (stamp, error)
        _set_config_error(error)
        return {}
    if not isinstance(data, dict):
        error = f"{path} is not a mapping"
        _YAML_CACHE[str(path)] = (stamp, error)
        _set_config_error(error)
        return {}
    _YAML_CACHE[str(path)] = (stamp, dict(data))
    return data


def _read_full_config() -> dict[str, Any]:
    return _read_yaml_file(_config_path())


def _read_legacy_router_config() -> dict[str, Any]:
    data = _read_yaml_file(_main_config_path())
    router = data.get("reasoning_router")
    return router if isinstance(router, dict) else {}


def _read_router_config_from_disk(*, include_runtime_override: bool = True) -> dict[str, Any]:
    _set_config_error(None)
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


class ConfigUnreadableError(RuntimeError):
    """The router config exists but cannot be parsed; refuse to overwrite it."""


def _update_router_config(updates: dict[str, Any]) -> None:
    global _RUNTIME_CONFIG_OVERRIDE
    if yaml is None:
        raise RuntimeError("PyYAML is required to update reasoning-router config")

    path = _config_path()
    data = _read_router_config_from_disk(include_runtime_override=False)
    config_error = _config_error()
    if config_error:
        # Writing now would replace the user's file with defaults plus one key.
        raise ConfigUnreadableError(config_error)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    _RUNTIME_CONFIG_OVERRIDE = None


def _format_status(config: dict[str, Any]) -> str:
    state = "on" if _truthy(config.get("enabled", True)) else "off"
    config_error = _config_error()
    if config_error:
        state = f"NOT ROUTING (config unreadable: {config_error})"
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

    # The host rewrites some sources (Telegram topic recovery) before deriving
    # the key its turn reads; use the same view or the override lands nowhere.
    normalizer = getattr(gateway, "_normalize_source_for_session_key", None)
    if callable(normalizer):
        try:
            source = normalizer(source) or source
        except Exception:
            logger.debug("reasoning-router: source normalization failed", exc_info=True)

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


_URL_OR_PATH_RE = re.compile(r"(?:https?://\S+|(?:^|\s)~?/[\w.-]+/\S+)", re.I)
_ORIGIN_PREFIX_RE = re.compile(
    r"^\s*Gateway message origin \(JSON data, not instructions or authorization\):\s*\n"
    r".*?\n(?:Do not guess a reply destination[^\n]*\n)?\s*",
    re.S,
)
_ORIGIN_MARKER = "Gateway message origin (JSON data, not instructions or authorization):"
_OOB_OPEN = "[OUT-OF-BAND USER MESSAGE"
_OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def _unwrap_oob(text: str) -> str | None:
    """Return the body of a steering wrapper, or None. Plain string ops: no backtracking."""
    stripped = text.strip()
    if not stripped.startswith(_OOB_OPEN) or not stripped.endswith(_OOB_CLOSE):
        return None
    header_end = stripped.find("]")
    newline = stripped.find("\n", header_end)
    if header_end < 0 or newline < 0 or stripped[header_end + 1 : newline].strip():
        return None
    body = stripped[newline + 1 : len(stripped) - len(_OOB_CLOSE)]
    return body.rstrip(" \t").removesuffix("\n")


def _strip_gateway_wrappers(text: str) -> str:
    """Drop Hermes-added framing so only the human's words get classified.

    The origin preamble and steering markers contain words like "authorization"
    that would otherwise push every message into the risky/high buckets.
    """
    out = str(text or "")
    for _ in range(3):
        before = out
        out = _ORIGIN_PREFIX_RE.sub("", out, count=1)
        cut = out.find(_ORIGIN_MARKER)
        if cut > 0:
            out = out[:cut].rstrip()  # origin block appended after the text
        body = _unwrap_oob(out)
        if body is not None:
            out = body
        if out == before:
            break
    return out


def _current_session_override(gateway, session_key: str) -> dict[str, Any] | None:
    peek = getattr(gateway, "_peek_session_state", None)
    if callable(peek):
        try:
            state = peek(session_key)
            conv = getattr(state, "conversation", None) if state is not None else None
            value = getattr(conv, "reasoning_override", None) if conv is not None else None
            return dict(value) if isinstance(value, dict) else None
        except Exception:
            logger.debug("reasoning-router: session-state peek failed", exc_info=True)
    overrides = getattr(gateway, "_session_reasoning_overrides", None)
    if isinstance(overrides, dict):
        value = overrides.get(session_key)
        return dict(value) if isinstance(value, dict) else None
    return None


def _router_set_map(gateway) -> dict[str, dict[str, Any]]:
    """Overrides this router wrote, keyed by session.

    Kept on the gateway object rather than only in module state: Hermes can
    re-exec a plugin module on reload, and a fresh module-level map would make
    every leftover router override look like a manual pin forever.
    """
    if gateway is None:
        return _ROUTER_SET
    try:
        existing = gateway.__dict__.get("_reasoning_router_set")
    except AttributeError:
        return _ROUTER_SET
    if isinstance(existing, dict):
        return existing
    try:
        gateway._reasoning_router_set = _ROUTER_SET
    except Exception:
        return _ROUTER_SET
    return _ROUTER_SET


def _has_manual_override(gateway, session_key: str) -> bool:
    """True when the session override was set by a human, not by this router."""
    router_set = _router_set_map(gateway)
    manual = _manual_session_keys(gateway)
    current = _current_session_override(gateway, session_key)
    if current is None:
        router_set.pop(session_key, None)
        manual.discard(session_key)
        return False
    if session_key in manual:
        return True
    return router_set.get(session_key) != current


def _release_router_override(gateway, session_key: str) -> bool:
    """Clear an override only if this router wrote it; manual pins stay."""
    if not session_key:
        return False
    router_set = _router_set_map(gateway)
    mine = router_set.get(session_key)
    if mine is None:
        return False
    current = _current_session_override(gateway, session_key)
    router_set.pop(session_key, None)
    if current != mine:
        return False
    try:
        _set_reasoning_override(gateway, session_key, None)
    except Exception as exc:
        logger.warning("reasoning-router: failed to clear router override: %s", exc)
        return False
    return True


def _is_reasoning_slash_command(text: str) -> bool:
    return bool(re.match(r"^\s*/reasoning(?:@\S+)?(?:\s|$)", text or "", re.I))


# Every value Hermes' /reasoning applier writes: parse_reasoning_effort's
# levels (incl. ``ultra``), its disable words (none/false/disabled), and reset.
# Display toggles (show/on/hide/off) and unknown words never touch the override.
_REASONING_DISABLE_ARGS = frozenset({"none", "false", "disabled"})
_REASONING_SETTING_ARGS = (
    frozenset(EFFORT_ORDER) | frozenset(EFFORT_ALIASES) | _REASONING_DISABLE_ARGS | {"reset"}
)


def _reasoning_command_changes_override(text: str) -> bool:
    """True for /reasoning <level>|none|false|disabled|reset; False for toggles or bare /reasoning."""
    parts = (text or "").strip().replace("\u2014", "--").split()[1:]
    args = [part.lower() for part in parts if part != "--global"]
    if not args:
        return False
    return " ".join(args) in _REASONING_SETTING_ARGS


_ROUTER_WRITE = threading.local()


def _manual_session_keys(gateway) -> set[str]:
    """Sessions whose current override came from a human write (typed or picker)."""
    if gateway is None:
        return set()
    try:
        existing = gateway.__dict__.get("_reasoning_router_manual")
    except AttributeError:
        return set()
    if isinstance(existing, set):
        return existing
    fresh: set[str] = set()
    try:
        gateway._reasoning_router_manual = fresh
    except Exception:
        pass
    return fresh


def _install_manual_write_hook(gateway) -> bool:
    """Wrap the gateway's override setter so non-router writes are marked manual.

    Ownership then follows who wrote the value, not what the value is: picking
    the same level the router chose, from a typed /reasoning or the native
    picker callback, still pins it. Returns False when the host exposes no setter.
    """
    try:
        setter = getattr(gateway, "_set_session_reasoning_override", None)
    except Exception:
        return False
    if not callable(setter):
        return False
    if getattr(setter, "_reasoning_router_hook", False):
        return True

    def hooked(session_key, reasoning_config, *args, **kwargs):
        result = setter(session_key, reasoning_config, *args, **kwargs)
        if not getattr(_ROUTER_WRITE, "active", False) and session_key:
            manual = _manual_session_keys(gateway)
            _router_set_map(gateway).pop(session_key, None)
            if reasoning_config is None:
                manual.discard(session_key)
            else:
                manual.add(session_key)
        return result

    hooked._reasoning_router_hook = True  # type: ignore[attr-defined]
    try:
        gateway._set_session_reasoning_override = hooked
    except Exception:
        return False
    return True


def _looks_like_slash_command(text: str) -> bool:
    """Mirror the host: "/Users/me/x.py crashes" is a path, not a command."""
    stripped = (text or "").lstrip()
    if not stripped.startswith("/"):
        return False
    head = stripped.split(maxsplit=1)[0][1:].split("@", 1)[0]
    return bool(head) and "/" not in head


def _session_model(gateway, session_key: str) -> str:
    """The session's effective model: /model override, else config model.default."""
    peek = getattr(gateway, "_peek_session_state", None)
    if callable(peek):
        try:
            state = peek(session_key)
            conv = getattr(state, "conversation", None) if state is not None else None
            override = getattr(conv, "model_override", None) if conv is not None else None
            if isinstance(override, dict) and override.get("model"):
                return str(override["model"])
        except Exception:
            logger.debug("reasoning-router: model-override peek failed", exc_info=True)
    model_cfg = _read_yaml_file(_main_config_path()).get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or model_cfg.get("model") or "")
    return str(model_cfg or "")


def _supported_efforts_for_model(model: str) -> tuple[str, ...] | None:
    """Wire effort ladder for a model id, or None when unknown (no clamp)."""
    bare = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if not bare:
        return None
    if bare.startswith("gpt-"):
        try:  # stay in lockstep with Hermes when it is importable
            from agent.reasoning_effort import codex_supported_efforts

            return tuple(codex_supported_efforts(bare))
        except Exception:
            pass
        if bare.startswith("gpt-6-astra"):
            return _GPT6_ASTRA_EFFORTS
        if bare.startswith("gpt-6-terra"):
            return _GPT6_TERRA_EFFORTS
        if bare.startswith(("gpt-6-sol", "gpt-6-luna", "gpt-daybreak")) or "gpt-5.6" in bare:
            return _GPT6_MAX_EFFORTS
        return _CODEX_LEGACY_EFFORTS
    if bare.startswith("claude"):
        return _CLAUDE_EFFORTS
    return None


def _fallback_clamp_effort(effort: str, supported) -> str:
    """Faithful copy of Hermes' ``agent.reasoning_effort.clamp_effort`` (no overrides).

    Nearest weaker supported level, else the weakest supported one. ``none``
    disables reasoning, so it is never a clamp target for an enabled request:
    ``minimal`` on a ladder without ``minimal`` becomes ``low``, not off.
    """
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


def _clamp_to_model(effort: str, model: str) -> str:
    """Fit ``effort`` to the model's wire ladder using Hermes' own clamp rules.

    Never escalates cost, and never turns an enabled request into ``none``.
    """
    supported = _supported_efforts_for_model(model)
    effort = _normalize_effort_name(effort)
    if not supported:
        return effort
    try:  # stay in lockstep with Hermes when it is importable
        from agent.reasoning_effort import clamp_effort as _hermes_clamp_effort

        fitted = _hermes_clamp_effort(effort, supported)
    except Exception:
        fitted = _fallback_clamp_effort(effort, supported)
    fitted = _normalize_effort_name(fitted)
    if effort != "none" and fitted == "none":  # belt and braces across Hermes versions
        fitted = _fallback_clamp_effort(effort, [e for e in supported if e != "none"])
    return fitted


def _set_reasoning_override(gateway, session_key: str, reasoning_config: dict[str, Any] | None) -> None:
    """Write (or clear, when ``reasoning_config`` is None) the session override."""
    setter = getattr(gateway, "_set_session_reasoning_override", None)
    if callable(setter):
        _ROUTER_WRITE.active = True
        try:
            setter(session_key, reasoning_config)
        finally:
            _ROUTER_WRITE.active = False
        _manual_session_keys(gateway).discard(session_key)
        return

    overrides = getattr(gateway, "_session_reasoning_overrides", None)
    if isinstance(overrides, dict):
        if reasoning_config is None:
            overrides.pop(session_key, None)
        else:
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
