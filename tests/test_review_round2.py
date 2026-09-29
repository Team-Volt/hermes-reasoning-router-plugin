"""Regression tests for the second review round on the GPT-6 routing PR."""

from __future__ import annotations

import pytest
from test_gpt6_overrides import KEY, StatefulGateway
from test_reasoning_router import FakeSessionStore, event, load_plugin

SID = "session-1"


def _route(plugin, gateway, text):
    plugin.pre_gateway_dispatch(event(text), gateway=gateway, session_store=FakeSessionStore(KEY, SID))
    return gateway._conv(KEY).reasoning_override


def _write_cfg(plugin, text):
    path = plugin._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# 1. clamp never disables an enabled minimum ---------------------------------


@pytest.mark.parametrize("model", ("gpt-6-sol", "gpt-6-luna", "gpt-6-terra", "claude-sonnet-5-5", "gpt-5.4"))
def test_minimal_clamps_to_low_not_none(model):
    plugin = load_plugin()
    assert plugin._clamp_to_model("minimal", model) == "low"


def test_fallback_clamp_matches_hermes_rules():
    plugin = load_plugin()
    ladder = ("none", "low", "medium", "high", "xhigh", "max")
    assert plugin._fallback_clamp_effort("minimal", ladder) == "low"
    assert plugin._fallback_clamp_effort("none", ladder) == "none"
    assert plugin._fallback_clamp_effort("max", ("none", "low", "high")) == "high"
    assert plugin._fallback_clamp_effort("none", ("low", "high")) == "low"
    assert plugin._fallback_clamp_effort("custom-tier", ladder) == "custom-tier"


def test_fallback_path_used_when_hermes_clamp_missing(monkeypatch):
    import sys

    plugin = load_plugin()
    monkeypatch.setitem(sys.modules, "agent.reasoning_effort", None)
    assert plugin._clamp_to_model("minimal", "gpt-6-sol") == "low"


def test_enabled_minimum_never_becomes_disabled_override():
    plugin = load_plugin()
    _write_cfg(plugin, "enabled: true\nmin: minimal\n")
    gateway = StatefulGateway(model="gpt-6-sol")
    for text in ("thanks", "ok", "what time is it", "lol nice"):
        override = _route(plugin, gateway, text)
        assert override is not None
        assert override.get("enabled") is not False, text
        assert override.get("effort") == "low", text


# 2. cheap informational routes need the whole message to qualify -------------


def test_definition_followed_by_action_is_not_cheap():
    plugin = load_plugin()
    effort, _ = plugin.classify_message("What does OAuth mean? Rotate the API keys", {"max": "max"})
    assert effort == "xhigh"


def test_status_question_followed_by_action_is_not_medium():
    plugin = load_plugin()
    effort, _ = plugin.classify_message(
        "what is the status of my server and then delete the old logs?", {"max": "max"}
    )
    assert effort in {"high", "xhigh"}


@pytest.mark.parametrize("text", ("What does OAuth mean?", "what is the status of my server?"))
def test_pure_informational_messages_keep_cheap_route(text):
    plugin = load_plugin()
    effort, _ = plugin.classify_message(text, {"max": "max"})
    assert effort in {"low", "medium"}


# 3. manual ownership follows who wrote, not what was written -----------------


def _router_pick(plugin, gateway):
    picked = _route(plugin, gateway, "ok cool")
    assert picked is not None and gateway._reasoning_router_set[KEY] == picked
    return dict(picked)


def test_equal_valued_typed_pin_survives_next_message():
    plugin = load_plugin()
    gateway = StatefulGateway()
    picked = _router_pick(plugin, gateway)
    plugin.pre_gateway_dispatch(event(f"/reasoning {picked['effort']}"), gateway=gateway, session_store=None)
    gateway._set_session_reasoning_override(KEY, dict(picked))  # Hermes' command handler writes it
    _route(plugin, gateway, "Audit the production deploy pipeline and root cause the outage.")
    assert gateway._conv(KEY).reasoning_override == picked


def test_equal_valued_picker_pin_survives_next_message():
    """Native picker callbacks never pass through pre_gateway_dispatch."""
    plugin = load_plugin()
    gateway = StatefulGateway()
    picked = _router_pick(plugin, gateway)
    gateway._set_session_reasoning_override(KEY, dict(picked))  # picker callback write
    assert plugin._has_manual_override(gateway, KEY)
    _route(plugin, gateway, "Audit the production deploy pipeline and root cause the outage.")
    assert gateway._conv(KEY).reasoning_override == picked


@pytest.mark.parametrize("arg", ("none", "false", "disabled", "ultra", "minimal", "max", "reset", "--global high"))
def test_all_hermes_reasoning_values_count_as_changes(arg):
    plugin = load_plugin()
    assert plugin._reasoning_command_changes_override(f"/reasoning {arg}")


@pytest.mark.parametrize("arg", ("show", "hide", "on", "off", "status", "banana"))
def test_display_toggles_and_unknown_args_do_not(arg):
    plugin = load_plugin()
    assert not plugin._reasoning_command_changes_override(f"/reasoning {arg}")


@pytest.mark.parametrize("arg", ("disabled", "false", "none"))
def test_manual_disable_after_router_none_is_kept(arg):
    plugin = load_plugin()
    _write_cfg(plugin, "enabled: true\nmin: none\n")
    gateway = StatefulGateway()
    gateway._conv(KEY).reasoning_override = None
    plugin._set_reasoning_override(gateway, KEY, {"enabled": False})
    plugin._router_set_map(gateway)[KEY] = {"enabled": False}
    plugin.pre_gateway_dispatch(event(f"/reasoning {arg}"), gateway=gateway, session_store=None)
    gateway._set_session_reasoning_override(KEY, {"enabled": False})
    _route(plugin, gateway, "Rotate every production API key and migrate the auth database.")
    assert gateway._conv(KEY).reasoning_override == {"enabled": False}


def test_reset_hands_session_back_to_router():
    plugin = load_plugin()
    gateway = StatefulGateway()
    _router_pick(plugin, gateway)
    gateway._set_session_reasoning_override(KEY, {"enabled": True, "effort": "high"})
    gateway._set_session_reasoning_override(KEY, None)  # /reasoning reset
    assert not plugin._has_manual_override(gateway, KEY)
    assert _route(plugin, gateway, "thanks") is not None


# 4. negation masks only its own clause ---------------------------------------


@pytest.mark.parametrize(
    "text",
    (
        "Do not restart the gateway but rotate the API keys",
        "Don't restart the gateway and rotate the API keys",
        "Don't restart the gateway, then rotate the API keys",
        "Do not restart the gateway; however rotate the API keys",
        "Don't restart the gateway. Rotate the API keys",
    ),
)
def test_positive_clause_after_negation_still_counts(text):
    plugin = load_plugin()
    effort, _ = plugin.classify_message(text, {"max": "max"})
    assert effort == "xhigh", text


def test_negated_risk_alone_still_masked():
    plugin = load_plugin()
    effort, _ = plugin.classify_message("don't wipe the drive, just show smart status", {"max": "max"})
    assert effort in {"low", "medium"}


# 5. quoted and fenced text is evidence, unless asked to run it ----------------


@pytest.mark.parametrize(
    "text",
    (
        'The log says "terraform destroy"',
        "What does this output mean?\n```\n$ terraform destroy\nDestroy complete! Resources: 3 destroyed.\n```",
        "the job printed `rm -rf /var/lib/app` in its output, why?",
    ),
)
def test_quoted_or_fenced_command_is_not_execution(text):
    plugin = load_plugin()
    effort, _ = plugin.classify_message(text, {"max": "max"})
    assert effort in {"low", "medium", "high"}, text
    assert effort != "xhigh"


@pytest.mark.parametrize(
    "text",
    (
        'Please run "terraform destroy" on staging',
        "execute this:\n```\nterraform destroy -auto-approve\n```",
        "go ahead and run `rm -rf /var/lib/app`",
    ),
)
def test_request_to_run_quoted_command_still_escalates(text):
    plugin = load_plugin()
    effort, _ = plugin.classify_message(text, {"max": "max"})
    assert effort == "xhigh", text


# 6. several explicit directives: strongest wins, limits still apply -----------


def test_strongest_explicit_directive_wins():
    plugin = load_plugin()
    text = "Use maximum reasoning and think hard about this"
    assert plugin.classify_message(text, {"max": "max"})[0] == "max"
    assert plugin.classify_message(text, {"max": "xhigh"})[0] == "xhigh"
    assert plugin._clamp_to_model("max", "gpt-6-terra") == "xhigh"


def test_think_hard_alone_is_still_xhigh():
    plugin = load_plugin()
    assert plugin.classify_message("think hard about this bug", {"max": "max"})[0] == "xhigh"


# 7. non-mapping configs are errors, not defaults -----------------------------


@pytest.mark.parametrize("body", ("[]\n", "false\n", "0\n", "- enabled\n", "just a string\n"))
def test_non_mapping_config_refuses_to_route(body):
    plugin = load_plugin()
    _write_cfg(plugin, body)
    gateway = StatefulGateway()
    assert _route(plugin, gateway, "Audit the production deploy pipeline.") is None
    assert plugin._config_error()
    assert "NOT ROUTING" in plugin._format_status(plugin._router_config(gateway))


@pytest.mark.parametrize("body", ("[]\n", "false\n", "0\n"))
def test_router_command_will_not_overwrite_non_mapping_config(body):
    plugin = load_plugin()
    path = _write_cfg(plugin, body)
    with pytest.raises(plugin.ConfigUnreadableError):
        plugin._update_router_config({"enabled": True})
    assert path.read_text() == body


def test_empty_config_file_still_means_defaults():
    plugin = load_plugin()
    _write_cfg(plugin, "")
    assert plugin._read_yaml_file(plugin._config_path()) == {}
    assert not plugin._config_error()
