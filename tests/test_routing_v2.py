"""Routing fixes found by replaying real chat traffic, plus override lifecycle fixes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from test_gpt6_overrides import KEY, StatefulGateway
from test_reasoning_router import FakeSessionStore, event, load_plugin

SID = "session-1"


def classify(text, **cfg):
    return load_plugin().classify_message(text, cfg)[0]


# --- classifier -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    (
        "why is the media server buffering on the living room tv?",
        "the bot replies twice to every message now",
        "jellyfin keeps crashing after the update",
        "whats wrong with the backup",
        "sonarr isn't picking up new episodes",
    ),
)
def test_short_troubleshooting_is_never_low(text):
    assert classify(text) in {"medium", "high"}


@pytest.mark.parametrize(
    "text",
    (
        "can we look into adding a memory plugin and compare it against built in memory?",
        "which ssd is better for a nas cache, nvme or sata",
        "is it worth switching the proxy to caddy",
    ),
)
def test_research_and_compare_is_never_low(text):
    assert classify(text) in {"medium", "high"}


@pytest.mark.parametrize(
    "text",
    (
        "migrate the app database to the new host with zero downtime and a rollback plan",
        "rotate the api keys on every server",
        "drop the old users table its not needed anymore",
        "force push the rewritten history to main",
    ),
)
def test_irreversible_or_fleet_wide_changes_are_xhigh(text):
    assert classify(text) == "xhigh"


def test_question_about_a_risky_change_is_high_not_xhigh():
    assert classify("what happens if i force push to main?") == "high"


@pytest.mark.parametrize(
    "text",
    (
        "how many tokens per second does the new model do",
        "permission denied when i open the share",
    ),
)
def test_risk_lookalike_phrases_are_masked(text):
    assert classify(text) != "xhigh"


@pytest.mark.parametrize(
    "text",
    (
        "write a script that prunes docker images older than 30 days",
        "write me a launchd plist that runs the sync every hour",
        "make the bot post a daily digest in the ops channel at 8am",
    ),
)
def test_build_requests_route_high(text):
    assert classify(text) == "high"


@pytest.mark.parametrize("text", ("what does oauth mean", "define idempotent"))
def test_definitions_stay_cheap(text):
    assert classify(text) == "low"


@pytest.mark.parametrize("text", ("thanks!", "lol nice", "ok cool", "👍"))
def test_closers_stay_low(text):
    assert classify(text) in {"none", "low"}


@pytest.mark.parametrize("text", ("2", "option b", "the first one", "continue", "keep going"))
def test_option_picks_and_continue_are_medium(text):
    assert classify(text) == "medium"


def test_curly_apostrophe_is_folded():
    assert classify("the tv app isn’t loading") in {"medium", "high"}


def test_troubleshoot_floor_is_configurable():
    assert classify("the bot replies twice to every message now", troubleshoot_floor="high") == "high"


# --- momentum -------------------------------------------------------------


def _history(tool_calls: int):
    history = [{"role": "user", "content": "do the thing"}]
    for i in range(tool_calls):
        history.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    history.append({"role": "assistant", "content": "Done."})
    return history


def _route(plugin, text, gateway, store):
    plugin.pre_gateway_dispatch(event(text), gateway=gateway, session_store=store)
    config = gateway.calls[-1][1]
    return config.get("effort", "none") if config.get("enabled", True) else "none"


def test_terse_followup_after_heavy_turn_inherits_weight():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(20), platform="discord")
    assert _route(plugin, "same for the other one", gateway, store) in {"medium", "high"}
    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(20), platform="discord")
    assert _route(plugin, "continue", gateway, store) == "high"


def test_closer_after_heavy_turn_stays_low():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(30), platform="discord")
    assert _route(plugin, "thanks!", gateway, store) in {"none", "low"}


def test_momentum_ignores_light_turns_and_expires():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(1), platform="discord")
    assert _route(plugin, "ok and the other one", gateway, store) == "low"

    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(20), platform="discord")
    plugin._MOMENTUM[SID]["at"] = datetime.now(UTC) - timedelta(hours=2)
    assert _route(plugin, "ok and the other one", gateway, store) == "low"


def test_turn_tool_calls_only_counts_the_last_turn():
    plugin = load_plugin()
    history = _history(9) + [{"role": "user", "content": "next"}] + _history(2)[1:]
    assert plugin._turn_tool_calls(history) == 2


# --- config safety --------------------------------------------------------


def test_unreadable_config_refuses_to_route(tmp_path):
    plugin = load_plugin()
    path = tmp_path / "reasoning-router" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("enabled: [unclosed\n")
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway.calls == []


def test_missing_yaml_parser_refuses_to_route(tmp_path, monkeypatch):
    plugin = load_plugin()
    path = tmp_path / "reasoning-router" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("enabled: true\n")
    monkeypatch.setattr(plugin, "yaml", None)
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway.calls == []


def test_config_recovers_once_file_is_fixed(tmp_path):
    plugin = load_plugin()
    path = tmp_path / "reasoning-router" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("enabled: [unclosed\n")
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    path.write_text("enabled: true\n")
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert len(gateway.calls) == 1


# --- override lifecycle ---------------------------------------------------


def test_shadow_mode_clears_a_leftover_router_override(tmp_path):
    plugin = load_plugin()
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway._conv(KEY).reasoning_override is not None
    cfg = tmp_path / "reasoning-router" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("shadow_mode: true\n")
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway._conv(KEY).reasoning_override is None


def test_disabling_the_router_clears_its_override_but_not_a_manual_pin(tmp_path):
    plugin = load_plugin()
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    cfg = tmp_path / "reasoning-router" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("enabled: false\n")
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway._conv(KEY).reasoning_override is None

    gateway._conv(KEY).reasoning_override = {"enabled": True, "effort": "xhigh"}
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    assert gateway._conv(KEY).reasoning_override == {"enabled": True, "effort": "xhigh"}


def test_reasoning_command_marks_the_next_value_as_manual_even_if_equal():
    plugin = load_plugin()
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(event("ok cool"), gateway=gateway, session_store=None)
    picked = dict(gateway._conv(KEY).reasoning_override)
    plugin.pre_gateway_dispatch(event(f"/reasoning {picked['effort']}"), gateway=gateway, session_store=None)
    gateway._conv(KEY).reasoning_override = dict(picked)  # the command itself runs after the hook
    assert plugin._has_manual_override(gateway, KEY)


def test_router_ownership_survives_a_plugin_reload():
    first = load_plugin()
    gateway = StatefulGateway()
    first.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=None)
    second = load_plugin()
    assert not second._has_manual_override(gateway, KEY)
    second.pre_gateway_dispatch(event("root cause the production outage and audit the deploy"), gateway=gateway,
                                session_store=None)
    assert gateway._conv(KEY).reasoning_override["effort"] in {"high", "xhigh"}


def test_message_starting_with_a_path_is_routed_not_treated_as_command():
    plugin = load_plugin()
    gateway = StatefulGateway()
    plugin.pre_gateway_dispatch(
        event("/srv/app/server.py crashes with a race condition in the auth migration, fix it"),
        gateway=gateway,
        session_store=None,
    )
    assert gateway.calls and gateway.calls[-1][1]["effort"] in {"high", "xhigh"}


def test_yaml_reads_are_cached_until_the_file_changes(tmp_path, monkeypatch):
    plugin = load_plugin()
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n")
    calls = []
    real = plugin.yaml.safe_load
    monkeypatch.setattr(plugin.yaml, "safe_load", lambda text: calls.append(1) or real(text))
    assert plugin._read_yaml_file(path) == {"a": 1}
    assert plugin._read_yaml_file(path) == {"a": 1}
    assert len(calls) == 1
    path.write_text("a: 22\n")
    assert plugin._read_yaml_file(path) == {"a": 22}
